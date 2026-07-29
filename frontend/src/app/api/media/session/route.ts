import { NextRequest, NextResponse } from "next/server"

export const dynamic = "force-dynamic"

const MEDIA_COOKIE = "hawkeye_media_access"
const MAX_TOKEN_LENGTH = 8192

function bearerToken(request: NextRequest): string | null {
  const authorization = request.headers.get("authorization")?.trim() || ""
  const match = authorization.match(/^Bearer ([^\s]+)$/i)
  const token = match?.[1]?.trim() || ""
  return token && token.length <= MAX_TOKEN_LENGTH ? token : null
}

export async function POST(request: NextRequest) {
  const token = bearerToken(request)
  if (!token) {
    return NextResponse.json(
      { success: false, error: "authentication required" },
      { status: 401, headers: { "Cache-Control": "no-store" } },
    )
  }

  const response = NextResponse.json(
    { success: true },
    { headers: { "Cache-Control": "no-store", "Referrer-Policy": "no-referrer" } },
  )
  response.cookies.set({
    name: MEDIA_COOKIE,
    value: token,
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "strict",
    path: "/api/media",
    maxAge: 15 * 60,
  })
  return response
}
