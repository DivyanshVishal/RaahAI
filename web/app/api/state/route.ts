import { getState } from "@/lib/store";

export const dynamic = "force-dynamic";

export function GET() {
  return Response.json(getState(), { headers: { "Cache-Control": "no-store" } });
}
