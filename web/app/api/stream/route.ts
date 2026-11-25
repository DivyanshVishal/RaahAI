import { getState, subscribe } from "@/lib/store";

export const dynamic = "force-dynamic";

export function GET(request: Request) {
  const encoder = new TextEncoder();
  let cleanup = () => {};
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      let closed = false;
      function send(frame: string) {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(frame));
        } catch {
          cleanup();
        }
      }
      const unsubscribe = subscribe((message) => {
        send(`event: ${message.type}\ndata: ${JSON.stringify(message)}\n\n`);
      });
      const heartbeat = setInterval(() => send(": heartbeat\n\n"), 15_000);
      cleanup = () => {
        if (closed) return;
        closed = true;
        clearInterval(heartbeat);
        unsubscribe();
        request.signal.removeEventListener("abort", abort);
      };
      function abort() {
        if (closed) return;
        cleanup();
        controller.close();
      }
      request.signal.addEventListener("abort", abort, { once: true });
      if (request.signal.aborted) return abort();
      send(`retry: 1500\nevent: snapshot\ndata: ${JSON.stringify(getState())}\n\n`);
    },
    cancel() {
      cleanup();
    },
  });
  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
