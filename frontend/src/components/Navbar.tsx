import React, { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Eye, Activity, Cpu } from 'lucide-react';
import { fetchHealth } from '../lib/api';
import { HealthStatus } from '../types';

export const Navbar: React.FC = () => {
  const location = useLocation();
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const navLinks = [
    { name: 'Studio', path: '/studio' },
    { name: 'Benchmarks', path: '/benchmarks' },
    { name: 'Methodology', path: '/methodology' },
    { name: 'About', path: '/about' },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#0A0D14]/90 backdrop-blur-md border-b border-[rgba(148,163,184,0.16)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-lg bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400 group-hover:bg-indigo-500/20 transition-all">
            <Eye className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold tracking-wider text-sm text-[#F3F6FC]">CVA</span>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">LAB</span>
            </div>
            <p className="text-[11px] text-[#718096]">Visual Reliability Lab</p>
          </div>
        </Link>

        {/* Links */}
        <nav className="hidden md:flex items-center gap-1">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.path;
            return (
              <Link
                key={link.path}
                to={link.path}
                className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-all ${
                  isActive
                    ? 'text-[#F3F6FC] bg-[#1B2638] border border-[rgba(148,163,184,0.16)]'
                    : 'text-[#A7B1C2] hover:text-[#F3F6FC] hover:bg-[#111827]'
                }`}
              >
                {link.name}
              </Link>
            );
          })}
        </nav>

        {/* Right Status & CTA */}
        <div className="flex items-center gap-3">
          {/* Health indicator */}
          <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-md bg-[#111827] border border-[rgba(148,163,184,0.16)] text-[11px] font-mono">
            <span className={`w-2 h-2 rounded-full ${health?.status === 'ok' ? 'bg-[#34D399] animate-pulse' : 'bg-[#FB7185]'}`} />
            <span className="text-[#A7B1C2]">
              {health?.status === 'ok' ? (
                <span className="flex items-center gap-1">
                  <Cpu className="w-3 h-3 text-indigo-400" />
                  {health.cuda_available ? (health.gpu_name?.split(' ')[1] || 'GPU') : 'CPU'}
                </span>
              ) : 'Offline'}
            </span>
          </div>

          <a
            href="https://github.com/Sambarlasagna/Counterfactual-Vision-Auditor"
            target="_blank"
            rel="noreferrer"
            className="p-2 rounded-md text-[#A7B1C2] hover:text-[#F3F6FC] hover:bg-[#111827] border border-transparent hover:border-[rgba(148,163,184,0.16)] transition-all text-xs font-mono"
            title="GitHub Repository"
          >
            GitHub
          </a>

          <Link
            to="/studio"
            className="hidden sm:inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-md bg-[#818CF8] hover:bg-[#6366F1] text-[#0A0D14] font-medium text-xs transition-all shadow-sm"
          >
            <Activity className="w-3.5 h-3.5" />
            Launch Studio
          </Link>
        </div>
      </div>
    </header>
  );
};
