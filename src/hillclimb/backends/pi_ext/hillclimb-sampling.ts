/** Inject per-operator sampling values supplied by the Hillclimb parent. */
export default function (pi: any) {
	const raw = process.env.HILLCLIMB_SAMPLING;
	if (!raw) return;

	let sampling: Record<string, number>;
	try {
		sampling = JSON.parse(raw);
	} catch (error) {
		throw new Error(`Invalid HILLCLIMB_SAMPLING JSON: ${String(error)}`);
	}
	if (!sampling || Array.isArray(sampling) || typeof sampling !== "object") {
		throw new Error("HILLCLIMB_SAMPLING must be a JSON object");
	}
	if (Object.values(sampling).some(value => typeof value !== "number" || !Number.isFinite(value))) {
		throw new Error("HILLCLIMB_SAMPLING values must be finite numbers");
	}
	if (Object.keys(sampling).length === 0) return;

	pi.on("before_provider_request", (event: any) => {
		console.error(`[hillclimb] sampling ${JSON.stringify(sampling)}`);
		return { ...event.payload, ...sampling };
	});
}
