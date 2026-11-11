import { SearchExperience } from "@/components/search-experience";
import { ProviderUsageIndicator } from "@/components/provider-usage";
import release from "@/lib/search/model-release.json";

export default function Home() {
  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Flightly home">
          <span className="brand-mark">FLY</span>
          <span>Flightly</span>
        </a>
        <ProviderUsageIndicator />
      </header>
      <SearchExperience />
      <footer>
        Base fares and ancillary estimates can change. Separate tickets are not protected
        connections.
        {" "}<a href={release.registryUrl} target="_blank" rel="noreferrer">Ranking model v{release.version}</a>
      </footer>
    </main>
  );
}
