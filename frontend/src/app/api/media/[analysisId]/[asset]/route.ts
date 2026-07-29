import { NextRequest, NextResponse } from "next/server"

export const dynamic = "force-dynamic"

const MEDIA_COOKIE = "hawkeye_media_access"
const ANALYSIS_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$/
const MEDIA_ASSETS = new Set([
  "skeleton_video",
  "original_video",
  "heatmap",
  "temporal_map",
  "attention_map",
  "overlay_video",
])
const FORWARDED_REQUEST_HEADERS = [
  "range",
  "if-range",
  "if-none-match",
  "if-modified-since",
]
const FORWARDED_RESPONSE_HEADERS = [
  "accept-ranges",
  "content-disposition",
  "content-length",
  "content-range",
  "content-type",
  "etag",
  "last-modified",
]

type MediaRouteContext = {
  params: Promise<{ analysisId: string; asset: string }>
}

function backendBaseUrl(): URL | null {
  const raw = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"
  try {
    const url = new URL(raw)
    if (!new Set(["http:", "https:"]).has(url.protocol)) return null
    // BACKEND_URL may include a trusted reverse-proxy mount such as
    // /hawkeye-api. Preserve that server-configured base path.
    if (!url.pathname.endsWith("/")) url.pathname = `${url.pathname}/`
    url.search = ""
    url.hash = ""
    return url
  } catch {
    return null
  }
}

async function proxyMedia(request: NextRequest, context: MediaRouteContext) {
  const { analysisId, asset } = await context.params
  if (
    !ANALYSIS_ID_PATTERN.test(analysisId)
    || analysisId.includes("..")
    || !MEDIA_ASSETS.has(asset)
  ) {
    return NextResponse.json(
      { success: false, error: "invalid media request" },
      { status: 400, headers: { "Cache-Control": "no-store" } },
    )
  }

  const accessToken = request.cookies.get(MEDIA_COOKIE)?.value
  if (!accessToken || accessToken.length > 8192) {
    return NextResponse.json(
      { success: false, error: "authentication required" },
      { status: 401, headers: { "Cache-Control": "no-store" } },
    )
  }

  const baseUrl = backendBaseUrl()
  if (!baseUrl) {
    return NextResponse.json(
      { success: false, error: "media service unavailable" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    )
  }

  const upstreamUrl = new URL(
    `api/analysis/media/${encodeURIComponent(analysisId)}/${encodeURIComponent(asset)}`,
    baseUrl,
  )
  const upstreamHeaders = new Headers({ Authorization: `Bearer ${accessToken}` })
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = request.headers.get(name)
    if (value) upstreamHeaders.set(name, value)
  }

  let upstream: Response
  try {
    upstream = await fetch(upstreamUrl, {
      method: request.method,
      headers: upstreamHeaders,
      cache: "no-store",
      redirect: "error",
    })
  } catch {
    return NextResponse.json(
      { success: false, error: "media service unavailable" },
      { status: 502, headers: { "Cache-Control": "no-store" } },
    )
  }

  const responseHeaders = new Headers({
    "Cache-Control": "no-store, private",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
  })
  for (const name of FORWARDED_RESPONSE_HEADERS) {
    const value = upstream.headers.get(name)
    if (value) responseHeaders.set(name, value)
  }

  return new NextResponse(request.method === "HEAD" ? null : upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  })
}

export function GET(request: NextRequest, context: MediaRouteContext) {
  return proxyMedia(request, context)
}

export function HEAD(request: NextRequest, context: MediaRouteContext) {
  return proxyMedia(request, context)
}
