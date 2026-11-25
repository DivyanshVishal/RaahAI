"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, FlaskConical, TrainFront } from "lucide-react";
import type { Message } from "@/lib/messages";
import { advance, initialState, type ReplayState } from "@/lib/replay-state";

const ReplayContext = createContext({ state: initialState, connected: false });
export function useReplay() { return useContext(ReplayContext); }

export function Shell({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ReplayState>(initialState);
  const [connected, setConnected] = useState(false);
  const current = usePathname();
  useEffect(() => {
    const stream = new EventSource("/api/stream");
    stream.onopen = () => setConnected(true);
    stream.onerror = () => setConnected(false);
    stream.addEventListener("snapshot", (event) => setState(JSON.parse(event.data)));
    for (const type of ["simulation_start", "batch", "simulation_end"]) {
      stream.addEventListener(type, (event) => {
        const message = JSON.parse(event.data) as Message;
        setState((previous) => advance(previous, message));
      });
    }
    return () => stream.close();
  }, []);

  const live = current !== "/predict";
  return (
    <ReplayContext.Provider value={{ state, connected }}>
      <header className="site-header">
        <div className="site-header-inner">
          <Link className="brand" href="/" aria-label="Rail Replay home">
            <span className="brand-icon"><TrainFront size={21} aria-hidden="true" /></span>
            <span>Rail Replay<small>Indian Railways</small></span>
          </Link>
          <nav aria-label="Main navigation">
            <Link className={live ? "nav-active" : ""} aria-current={current === "/" ? "page" : undefined} href="/">
              <Activity size={16} aria-hidden="true" /><span>Live dashboard</span>
            </Link>
            <Link className={!live ? "nav-active" : ""} aria-current={!live ? "page" : undefined} href="/predict">
              <FlaskConical size={16} aria-hidden="true" /><span>What-if prediction</span>
            </Link>
          </nav>
          <span className="connection" role="status"><i className={connected ? "online" : ""} />
            {connected ? "Connected" : "Connecting"}
          </span>
        </div>
      </header>
      <main className="page">
        {children}
        <footer className="footer"><TrainFront size={15} aria-hidden="true" /> Rail Replay <span>All times shown in IST</span></footer>
      </main>
    </ReplayContext.Provider>
  );
}
