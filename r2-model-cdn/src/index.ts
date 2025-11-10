export default {
  async fetch(request: Request, env: any): Promise<Response> {
    const url = new URL(request.url);
    const key = url.pathname.slice(1);

    const object = await env.BUCKET.get(key);
    if (!object) return new Response("Not Found", { status: 404 });

    return new Response(object.body, {
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': object.httpMetadata?.contentType || 'application/octet-stream'
      }
    });
  }
};
