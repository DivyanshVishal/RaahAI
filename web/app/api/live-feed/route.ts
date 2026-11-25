export const runtime = "nodejs";

export async function GET(request: Request) {
  const query = new URL(request.url).searchParams;
  try {
    const feed = new URL("/api/events", process.env.LIVE_FEED_URL || "http://127.0.0.1:8100");
    for (const key of ["train_no", "journey_date", "after", "run", "session"]) {
      const value = query.get(key);
      if (value) feed.searchParams.set(key, value);
    }
    const response = await fetch(feed, { cache: "no-store", signal: AbortSignal.timeout(query.get("run") ? 5000 : 30000) });
    return new Response(response.body, {
      status: response.status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch {
    return Response.json({ error: "The simulated feed could not be reached." }, { status: 503 });
  }
}
