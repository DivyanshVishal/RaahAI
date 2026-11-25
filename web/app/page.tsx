import Link from "next/link";
import { ArrowRight, Search } from "lucide-react";
import { RouteDetails } from "@/components/route-details";
import { directory, loadJourney, searchTrains, type JourneyRoute } from "@/lib/train-routes";

type Params = Record<string, string | string[] | undefined>;
const value = (param: Params[string]) => typeof param === "string" ? param.trim() : "";
const pageSize = 20;

export default async function Home({ searchParams }: { searchParams: Promise<Params> }) {
  const params = await searchParams;
  const query = value(params.q).slice(0, 100);
  const number = value(params.train);
  const date = value(params.date);
  const trains = await directory();
  const matches = query ? searchTrains(trains, query) : [];
  const selected = number ? trains.find((train) => train.number === number)
    : query ? (matches.length === 1 ? matches[0] : undefined)
    : trains.find((train) => train.number === "18636");
  const pages = Math.ceil(matches.length / pageSize);
  const page = Math.max(1, Math.min(pages, Number.parseInt(value(params.page), 10) || 1));
  let route: JourneyRoute | null = null;
  let error = "";
  if (selected) {
    try {
      route = !date || /^\d{4}-\d{2}-\d{2}$/.test(date) ? await loadJourney(selected.file, date) : null;
      if (!route) error = "There is no recorded journey on this date.";
    } catch (cause) {
      console.error("Could not load recorded journey", selected.number, cause);
      error = "The station records could not be loaded. Please try again.";
    }
  }
  const searchPage = (p: number) => "/?" + new URLSearchParams({ q: query, page: String(p) });
  return <>
    <section className="atlas-hero"><div className="atlas-wrap hero-content">
      <div><h1>Find your train route</h1><p>Explore recorded journeys, station stops and arrival times.</p></div>
      <form action="/" className="atlas-form">
        <label htmlFor="train-query">Train name or number</label>
        <div className="atlas-search"><Search size={20} aria-hidden="true" />
          <input id="train-query" key={query} name="q" type="search" required maxLength={100} defaultValue={query} placeholder="e.g. Ranchi Express or 18636" />
          <button type="submit">Search</button>
        </div>
      </form>
    </div></section>
    <div className="atlas-wrap atlas-body">
      {route && selected ? <RouteDetails key={selected.number + route.date} train={{ number: selected.number, name: selected.name }} route={route} />
        : selected ? <section className="atlas-message"><h2>{error}</h2><Link href={"/?train=" + selected.number}>Open latest recorded journey</Link></section>
        : matches.length && !number ? <section className="matches"><h2>Choose your train</h2>
          <p>{matches.length.toLocaleString("en-IN")} matches for “{query}”</p>
          <ul>{matches.slice((page - 1) * pageSize, page * pageSize).map((train) =>
            <li key={train.number}><Link href={"/?train=" + train.number} prefetch={false}>
              <span className="train-no">{train.number}</span><strong>{train.name || "Unnamed train"}</strong><ArrowRight size={17} />
            </Link></li>)}</ul>
          {pages > 1 && <nav aria-label="Search result pages">
            {page > 1 && <Link href={searchPage(page - 1)}>Previous</Link>}<span>Page {page} of {pages}</span>
            {page < pages && <Link href={searchPage(page + 1)}>Next</Link>}
          </nav>}
        </section> : <section className="atlas-message"><h2>No trains found</h2>
          <p>Try a train number or a shorter part of the train name.</p></section>}
    </div>
  </>;
}
