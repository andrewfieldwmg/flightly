import release from "@/lib/search/model-release.json";

export const dynamic = "force-static";

export function GET() {
  return Response.json(release);
}
