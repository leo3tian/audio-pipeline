import os
import boto3
from yt_dlp import YoutubeDL
import tqdm
import json

# --- Configuration ---
SQS_QUEUE_URL = os.getenv("AWS_QUEUE_DOWNLOAD")
if not SQS_QUEUE_URL:
    raise ValueError("Environment variable AWS_QUEUE_DOWNLOAD must be set")

# The list of YouTube channel URLs you want to process
CHANNEL_URLS = [
    "https://www.youtube.com/playlist?list=PLk1Sqn_f33KuWf3tW9BBe_4TP7x8l0m3T"
]

def send_messages_in_batches(sqs_client, messages, batch_size=10):
    """
    Send messages to SQS in batches of up to 10 (SQS limit).
    Returns the number of successfully sent messages.
    """
    sent_count = 0
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        entries = [
            {
                'Id': str(j),  # Batch message ID (must be unique within the batch)
                'MessageBody': json.dumps({
                    'video_id': msg['video_id'],
                    'url': msg['url']
                })
            }
            for j, msg in enumerate(batch)
        ]
        
        try:
            response = sqs_client.send_message_batch(
                QueueUrl=SQS_QUEUE_URL,
                Entries=entries
            )
            
            # Count successful sends
            sent_count += len(entries)
            
            # Handle any failed messages
            if 'Failed' in response and response['Failed']:
                sent_count -= len(response['Failed'])
                print("\n[!] Some messages failed to send:")
                for failed in response['Failed']:
                    print(f"  - Message {failed['Id']}: {failed['Message']}")
                    sent_count -= 1
                    
        except Exception as e:
            print(f"\n[!] Error sending batch: {str(e)}")
            continue
            
    return sent_count

def main():
    """
    Scans YouTube channels to find all video URLs and sends them to an SQS queue
    for processing. This only needs to be run once to kick off the pipeline.
    """
    print("0: Starting task setup: fetching all video URLs from channels...")
    sqs_client = boto3.client("sqs")
    all_videos = []  # List to store video information before sending

    # Use yt-dlp to extract all video URLs from the channels
    ydl_opts = {'extract_flat': True, 'quiet': True, 'ignoreerrors': True}
    with YoutubeDL(ydl_opts) as ydl:
        for channel_url in CHANNEL_URLS:
            print(f"  Fetching videos from: {channel_url}")
            try:
                info = ydl.extract_info(channel_url, download=False)
                if info and info.get("entries"):
                    for entry in info['entries']:
                        if entry and 'id' in entry:
                            video_id = entry['id']
                            url = f"https://www.youtube.com/watch?v={video_id}"
                            # Store both ID and URL for sending
                            all_videos.append({
                                'video_id': video_id,
                                'url': url
                            })
            except Exception as e:
                print(f"  [!] Could not process channel {channel_url}: {e}")

    video_count = len(all_videos)
    if video_count == 0:
        print("\n[!] No videos found. Aborting task setup.")
        return

    print(f"\nFound {video_count} unique videos.")
    print(f"Sending tasks to SQS queue: {SQS_QUEUE_URL}")
    
    # Send messages in batches and track progress
    sent_count = send_messages_in_batches(sqs_client, all_videos)
            
    print("\n✅ Video task setup complete.")
    print(f"   {sent_count}/{video_count} tasks sent to SQS queue")
    
    if sent_count < video_count:
        print(f"   [!] Warning: {video_count - sent_count} tasks failed to send")

if __name__ == "__main__":
    main()
