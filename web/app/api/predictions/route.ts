import { messageSchema } from "@/lib/messages";
import { getState, ingest } from "@/lib/store";

export async function POST(request: Request) {
  const body = await request.json().catch(() => undefined);
  const result = messageSchema.safeParse(body);
  if (!result.success) {
    return Response.json({ error: "Invalid simulator message", issues: result.error.issues.slice(0, 12) }, { status: 400 });
  }
  if (result.data.type !== "simulation_start" && !getState().start) {
    return Response.json({ error: "Send simulation_start first" }, { status: 409 });
  }
  ingest(result.data);
  return Response.json({ ok: true });
}
