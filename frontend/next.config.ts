import type { NextConfig } from "next";

const backendUrl = process.env.BACKEND_URL || "http://localhost:5000";

const nextConfig: NextConfig = {
  // Empty turbopack config to silence webpack migration warning
  turbopack: {},

  // Proxy backend requests to avoid CORS issues with video files
  async rewrites() {
    return {
      // Run these only after Next.js app routes (including /api/media) have
      // had a chance to match. Array-form rewrites run before dynamic routes.
      fallback: [
        {
          source: '/api/backend/:path*',
          destination: `${backendUrl}/api/:path*`,
        },
        {
          source: '/api/:path*',
          destination: `${backendUrl}/api/:path*`,
        },
        {
          source: '/files/:path*',
          destination: `${backendUrl}/files/:path*`,
        },
      ],
    }
  },
};

export default nextConfig;
