import { commit, deleteFiles } from '@huggingface/hub';

export interface Env {
	R2: R2Bucket;
	HF_REPO_ID: string;
	R2_BUCKET_NAME: string;
	HF_TOKEN: string;
	MAX_BYTES_PER_SHARD?: string;
	INWORKER_CONCURRENCY?: string;
	DRY_RUN?: string;
}

type PlanMessage = {
	plan_key: string;
	language: string;
	shard_id_prefix: string;
};

type EpisodePlan = string; // R2 prefix like processed/en/<episode_id>/

export default {
	async queue(batch: MessageBatch<any>, env: Env, ctx: ExecutionContext): Promise<void> {
		for (const msg of batch.messages) {
			try {
				// The enqueuer now sends a JSON string, so we must parse it.
				const body: PlanMessage = typeof msg.body === 'string' ? JSON.parse(msg.body) : msg.body;
				await handlePlanMessage(env, body);
				msg.ack();
			} catch (err) {
				console.error('Plan processing failed', err);
				msg.retry();
			}
		}
	},
} satisfies ExportedHandler<Env>;

async function handlePlanMessage(env: Env, body: PlanMessage) {
	const { plan_key, language, shard_id_prefix } = body;
	console.log(`[plan] processing ${plan_key} (${language})`);

	const planObj = await env.R2.get(plan_key);
	if (!planObj) throw new Error(`plan not found: ${plan_key}`);
	const text = await planObj.text();
	const episodePrefixes: EpisodePlan[] = text.split('\n').map(l => l.trim()).filter(Boolean);
	if (episodePrefixes.length === 0) {
		console.log('[plan] empty plan, skipping');
		return;
	}

	const maxBytes = parseInt(env.MAX_BYTES_PER_SHARD || '1000000000', 10);
	const concurrency = parseInt(env.INWORKER_CONCURRENCY || '8', 10);
	const dryRun = env.DRY_RUN === 'true';

	let bytesUploaded = 0;

	const episodeProcessingTasks = episodePrefixes.map(episodePrefix => async () => {
		if (bytesUploaded >= maxBytes) {
			return;
		}
		const { uploadedBytes } = await processOneEpisode(env, episodePrefix, language, dryRun);
		bytesUploaded += uploadedBytes;
	});

	await runWithConcurrencyLimit(episodeProcessingTasks, concurrency);

	if (bytesUploaded >= maxBytes) {
		console.log(`[plan] byte cap reached (${bytesUploaded} bytes), stopping early`);
	}

	console.log(`[plan] done ${plan_key}, uploadedBytes=${bytesUploaded}`);
}

// --- Hugging Face Create-Commit API Flow ---

interface HfFileCommit {
	path: string;
	size: number;
}

interface HfPresignedUrl {
	path: string;
	uploadUrl: string;
}

function createJsonContent(segment: any): string {
	return JSON.stringify({
		text: segment?.text ?? '',
		speaker_id: segment?.speaker ?? 'UNKNOWN',
		duration: (segment?.end ?? 0) - (segment?.start ?? 0),
		dnsmos: segment?.dnsmos ?? 0,
		language: segment?.language ?? 'UNKNOWN',
	});
}

async function processOneEpisode(env: Env, episodePrefix: string, language: string, dryRun: boolean) {
	// 1. Get metadata and prepare file list for commit
	const metaKey = `${episodePrefix}all_segments.json`;
	const metaObj = await env.R2.get(metaKey);
	if (!metaObj) {
		console.warn(`[episode] missing metadata: ${metaKey}`);
		return { uploadedBytes: 0 };
	}
	const metaText = await metaObj.text();
	let segments: any[] = [];
	try {
		segments = JSON.parse(metaText);
	} catch {
		console.warn(`[episode] bad metadata JSON: ${metaKey}`);
		return { uploadedBytes: 0 };
	}

	const filesToCommit: HfFileCommit[] = [];
	let totalBytes = 0;

	// Add JSON sidecar for each audio file to the commit plan
	for (let i = 0; i < segments.length; i++) {
		const episodeId = episodePrefix.split('/')[2];
		const baseName = `${episodeId}_${i.toString().padStart(6, '0')}`;
		const jsonPath = `data/${language}/${baseName}.json`;
		const segment = segments[i];
		const jsonContent = createJsonContent(segment);
		filesToCommit.push({ path: jsonPath, size: jsonContent.length });
		totalBytes += jsonContent.length;
	}

	// Add audio files to commit plan by getting their size from R2 head
	const audioFilePaths = await Promise.all(
		segments.map(async (_, i) => {
			const episodeId = episodePrefix.split('/')[2];
			const baseName = `${episodeId}_${i.toString().padStart(6, '0')}`;
			const r2AudioKey = `${episodePrefix}${baseName}.mp3`;
			const head = await env.R2.head(r2AudioKey);
			if (head) {
				totalBytes += head.size;
				return { path: `data/${language}/${baseName}.mp3`, size: head.size, r2Key: r2AudioKey };
			}
			return null;
		})
	);
	
	const validAudioFiles = audioFilePaths.filter(f => f !== null) as { path: string; size: number; r2Key: string }[];
	filesToCommit.push(...validAudioFiles);

	if (filesToCommit.length === 0) {
		console.log(`[episode] No valid files found for ${episodePrefix}`);
		return { uploadedBytes: 0 };
	}

	if (dryRun) {
		console.log(`[dry-run] Would commit ${filesToCommit.length} files for ${episodePrefix}`);
		return { uploadedBytes: totalBytes };
	}

	// 2. Use the HF Hub SDK to create the commit
	console.log(`[episode] Starting commit for ${episodePrefix} with ${filesToCommit.length} files...`);
	
	try {
		const operations = await Promise.all(filesToCommit.map(async (file) => {
			const isJson = file.path.endsWith('.json');
			if (isJson) {
				const baseName = file.path.split('/').pop()!.replace('.json', '');
				const segmentIndex = parseInt(baseName.split('_').pop()!, 10);
				const segment = segments[segmentIndex];
				const jsonContent = createJsonContent(segment);
				return {
					path: file.path,
					content: new Blob([jsonContent], { type: 'application/json' })
				};
			} else {
				const audioFileInfo = validAudioFiles.find(af => af.path === file.path);
				if (audioFileInfo) {
					const r2Object = await env.R2.get(audioFileInfo.r2Key);
					if (r2Object) {
						return {
							path: file.path,
							content: r2Object.body,
						};
					}
				}
			}
			return null;
		}));

		const validOperations = operations.filter(op => op !== null) as { path: string; content: Blob | ReadableStream }[];
		
		await commit({
			repo: { id: env.HF_REPO_ID, type: 'dataset' },
			credentials: { accessToken: env.HF_TOKEN },
			operations: validOperations,
			commitMessage: `Add episode ${episodePrefix}`,
			createPr: false,
		});

		console.log(`[episode] Successfully committed ${episodePrefix}`);
		return { uploadedBytes: totalBytes };

	} catch (err) {
		console.error(`HF commit failed for ${episodePrefix}:`, err);
		throw err; // Re-throw to fail the queue message and trigger a retry
	}
}

async function runWithConcurrencyLimit(tasks: Array<() => Promise<void>>, limit: number) {
	let inFlight = 0;
	let idx = 0;
	return await new Promise<void>((resolve, reject) => {
		const next = () => {
			if (idx >= tasks.length && inFlight === 0) return resolve();
			while (inFlight < limit && idx < tasks.length) {
				const current = tasks[idx++]!;
				inFlight++;
				current().catch((err) => {
					// A single failure will reject the whole batch for retry
					console.error('Upload task failed, rejecting batch', err);
					reject(err);
				}).finally(() => {
					inFlight--;
					next();
				});
			}
		};
		next();
	});
}


