import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Empty turbopack config to silence webpack migration warning
  turbopack: {},

  // Proxy backend requests to avoid CORS issues with video files
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: 'http://localhost:5000/api/:path*',
      },
      {
        source: '/files/:path*',
        destination: 'http://localhost:5000/files/:path*',
      },
    ]
  },
};

export default nextConfig;
