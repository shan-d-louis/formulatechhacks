const backendBaseUrl = Deno.env.get("BACKEND_BASE_URL");

Deno.serve(async (request) => {
  if (!backendBaseUrl) {
    return Response.json(
      { ok: false, error: "BACKEND_BASE_URL is not configured" },
      { status: 500 },
    );
  }

  const incoming = new URL(request.url);
  const prefix = "/sidewall-health";
  const path = incoming.pathname.startsWith(prefix)
    ? incoming.pathname.slice(prefix.length) || "/health"
    : incoming.pathname || "/health";
  const target = new URL(`${backendBaseUrl.replace(/\/$/, "")}${path}`);
  target.search = incoming.search;

  const upstream = await fetch(target, {
    method: request.method,
    headers: request.headers,
    body: request.body,
  });
  const body = await upstream.text();

  return new Response(body, {
    status: upstream.status,
    headers: {
      "content-type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
});
