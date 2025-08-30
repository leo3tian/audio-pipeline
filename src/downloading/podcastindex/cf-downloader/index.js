// cf-downloader/index.js

// This is the main entry point for our Cloudflare Worker.
// We will be using the ES Modules format.

// We'll need the AWS SDK for DynamoDB.
import { DynamoDBClient, PutItemCommand } from "@aws-sdk/client-dynamodb";

// --- Configuration ---
const AWS_REGION = "us-west-1";
const DYNAMODB_TABLE_NAME = "PodcastIndexJobs";
const FETCH_TIMEOUT_MS = 10000;

// This is the main handler for our worker.
export default {
    // The 'queue' handler is triggered by messages from our bound queue.
    async queue(batch, env) {
        //console.log(`Received a batch of ${batch.messages.length} messages.`);

        const awsCreds = {
            region: AWS_REGION,
            credentials: {
                accessKeyId: env.AWS_ACCESS_KEY_ID,
                secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
            },
        };
        const dynamoDbClient = new DynamoDBClient(awsCreds);

        // We can now process the batch concurrently again, as the streaming
        // implementation in processDownloadJob prevents memory issues.
        const promises = batch.messages.map(message =>
            processDownloadJob(message, env, dynamoDbClient)
        );
        
        // Use 'await Promise.allSettled' to ensure all jobs are attempted,
        // even if some fail.
        await Promise.allSettled(promises);
        
        // console.log("Finished processing batch.");
    },
};

async function processDownloadJob(message, env, dynamoDbClient) {
    const jobData = message.body;
    const { episode_url, podcast_id, language } = jobData;

    if (!episode_url) {
        console.error("Message is missing 'episode_url'", jobData);
        message.ack(); // Acknowledge to remove from queue
        return;
    }

    //console.log(`Processing job for: ${episode_url}`);

    try {
        // 1. Perform conditional write to DynamoDB to claim the job
        try {
            const putCommand = new PutItemCommand({
                TableName: DYNAMODB_TABLE_NAME,
                Item: { 'episode_url': { S: episode_url } },
                ConditionExpression: 'attribute_not_exists(episode_url)',
            });
            await dynamoDbClient.send(putCommand);
        } catch (error) {
            if (error.name === 'ConditionalCheckFailedException') {
                // console.log(`Duplicate job found: ${episode_url}. Acknowledging message.`);
                message.ack(); // Acknowledge to remove from queue
                return;
            }
            throw error;
        }

        // 2. Timeout Controller & Download
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
        
        const requestUrl = new URL(episode_url);
        const isBuzzsproutHost = requestUrl.hostname.endsWith("buzzsprout.com");
        const fetchOptions = { signal: controller.signal };
        if (!isBuzzsproutHost) {
            fetchOptions.headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' };
        }

        const response = await fetch(episode_url, fetchOptions);
        
        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        if (!response.body) {
            throw new Error('Response body is null. Cannot stream data.');
        }
        
        // 3. Save to R2 using a hybrid streaming/buffered approach.
        const r2Key = await getR2Key(podcast_id, language, episode_url);
        const httpMetadata = { contentType: response.headers.get('content-type') || 'audio/mpeg' };

        // R2's streaming upload requires the Content-Length header to be present.
        const contentLength = response.headers.get('Content-Length');

        if (contentLength) {
            // If the content length is known, we can stream the response directly.
            // This is the most memory-efficient method.
            await env.R2_BUCKET.put(r2Key, response.body, { httpMetadata });
        } else {
            // If the content length is not known (e.g., chunked encoding), we must
            // buffer the entire response into memory before uploading.
            console.warn(`No Content-Length for ${episode_url}. Falling back to buffered upload.`);
            const audioData = await response.arrayBuffer();
            await env.R2_BUCKET.put(r2Key, audioData, { httpMetadata });
        }
        
        // console.log(`Successfully saved ${r2Key} to R2.`);
        
        // 4. Acknowledge the message to remove it from the queue
        message.ack();
        // console.log(`Successfully processed job for ${episode_url}`);

    } catch (error) {
        console.error(error, `for ${episode_url}`);
        
        // --- Intelligent Retry Logic ---
        // Implement randomized delay (jitter) for retries to handle transient network issues.
        const maxRetries = 3; // Corresponds to the setting in wrangler.toml
        
        if (message.retryCount < maxRetries) {
            // Calculate a random delay between 1 and 5 minutes.
            const minDelaySeconds = 60;
            const maxDelaySeconds = 300;
            const delaySeconds = Math.floor(Math.random() * (maxDelaySeconds - minDelaySeconds + 1)) + minDelaySeconds;
            
            console.log(`Retrying job for ${episode_url} in ${delaySeconds} seconds (Attempt ${message.retryCount + 1}).`);
            message.retry({ delay: delaySeconds });
        } else {
            // If we've exhausted all retries, acknowledge the message to send it to the Dead Letter Queue.
            // console.error(`Job for ${episode_url} has failed after ${maxRetries} retries. Sending to DLQ.`);
            message.ack();
        }
    }
}

async function getR2Key(podcastId = 'unknown', language = 'unknown', episodeUrl) {
    const url = new URL(episodeUrl);
    let extension = url.pathname.split('.').pop() || 'mp3';
    if (extension.length > 5 || extension.includes('/')) {
        extension = 'mp3';
    }

    const encoder = new TextEncoder();
    const data = encoder.encode(episodeUrl);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);

    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    
    const shortHash = hashHex.substring(0, 16);

    return `raw_audio/${language}/${podcastId}/${shortHash}.${extension}`;
} 