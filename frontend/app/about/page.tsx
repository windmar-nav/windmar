'use client';

import Header from '@/components/Header';
import { Ship, Scale, AlertTriangle, Github, ExternalLink, Mail, Compass } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-maritime-darker text-white">
      <Header />
      <div className="container mx-auto px-6 pt-20 pb-12 max-w-3xl">

        {/* Title */}
        <div className="flex items-center space-x-3 mb-8">
          <Ship className="w-8 h-8 text-primary-400" />
          <div>
            <h1 className="text-3xl font-bold maritime-gradient-text">WINDMAR</h1>
            <p className="text-sm text-gray-400">Weather Routing & Performance Analytics</p>
          </div>
        </div>

        {/* About */}
        <section className="mb-10">
          <p className="text-gray-300 leading-relaxed">
            Windmar is a weather routing and vessel performance analytics platform
            designed for merchant ships. It combines physics-based fuel consumption
            modelling, multi-source weather data, and route optimization algorithms to
            support fuel-efficient voyage planning.
          </p>
          <div className="mt-4 flex flex-wrap gap-3 text-sm">
            <a
              href="https://github.com/windmar-nav/windmar"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-white/5 text-gray-300 hover:text-white hover:bg-white/10 transition-all"
            >
              <Github className="w-4 h-4" />
              <span>Source Code</span>
            </a>
            <a
              href="https://windmar-nav.github.io"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-white/5 text-gray-300 hover:text-white hover:bg-white/10 transition-all"
            >
              <ExternalLink className="w-4 h-4" />
              <span>Documentation</span>
            </a>
            <a
              href="mailto:contact@slmar.co"
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-white/5 text-gray-300 hover:text-white hover:bg-white/10 transition-all"
            >
              <Mail className="w-4 h-4" />
              <span>Contact</span>
            </a>
          </div>
        </section>

        <hr className="border-white/10 mb-10" />

        {/* Philosophy */}
        <section className="mb-10">
          <div className="flex items-center space-x-2 mb-4">
            <Compass className="w-5 h-5 text-primary-400" />
            <h2 className="text-xl font-semibold">Philosophy</h2>
          </div>
          <div className="space-y-3 text-sm text-gray-400 leading-relaxed">
            <p>
              Route analysis and performance monitoring tools belong on board, where
              decisions are made. The Master and bridge team should have direct access
              to the models, the data, and the reasoning behind every recommendation.
            </p>
            <p>
              Shore-side black boxes that deliver opaque instructions to the vessel
              undermine professional judgement. Commercial secrecy around the algorithms
              that influence safety-critical decisions is not a healthy position for the
              industry.
            </p>
            <p>
              Windmar is built as open-source software with transparent, physics-based
              models. Every calculation can be inspected, every assumption challenged,
              and every parameter calibrated against real operational data. The people
              who navigate the vessel should own and understand the tools they rely on.
            </p>
          </div>
        </section>

        <hr className="border-white/10 mb-10" />

        {/* Disclaimer */}
        <section className="mb-10">
          <div className="flex items-center space-x-2 mb-4">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            <h2 className="text-xl font-semibold">Disclaimer</h2>
          </div>
          <div className="space-y-3 text-sm text-gray-400 leading-relaxed">
            <p>
              <strong className="text-gray-300">Not a navigation aid.</strong> Windmar is a
              planning and analytics tool. It is not certified as an Electronic Chart Display
              and Information System (ECDIS) or any other IMO-approved navigation equipment.
              It must not be used for navigation or as a substitute for proper bridge
              watchkeeping procedures.
            </p>
            <p>
              <strong className="text-gray-300">No guarantee of accuracy.</strong> Weather
              forecasts, route optimizations, fuel consumption estimates, and CII projections
              are based on mathematical models and third-party data sources (NOAA GFS,
              Copernicus Marine Service, ERA5). Actual conditions may differ significantly.
              All results should be treated as indicative, not definitive.
            </p>
            <p>
              <strong className="text-gray-300">Professional judgement required.</strong> The
              Master and responsible officers retain full authority and responsibility for
              vessel navigation, safety, and compliance with SOLAS, COLREG, MARPOL, and all
              applicable regulations. Windmar outputs do not replace professional maritime
              judgement.
            </p>
            <p>
              <strong className="text-gray-300">No liability.</strong> To the maximum extent
              permitted by law, the authors and contributors accept no liability for any loss,
              damage, injury, or expense arising from the use of this software, including but
              not limited to vessel damage, cargo loss, environmental incidents, or personal
              injury.
            </p>
          </div>
        </section>

        <hr className="border-white/10 mb-10" />

        {/* License */}
        <section className="mb-10">
          <div className="flex items-center space-x-2 mb-4">
            <Scale className="w-5 h-5 text-primary-400" />
            <h2 className="text-xl font-semibold">License</h2>
          </div>
          <div className="bg-white/5 rounded-lg p-5 text-sm text-gray-400 leading-relaxed space-y-3">
            <p>
              Copyright 2024-2026{' '}
              <a href="https://slmar.co" target="_blank" rel="noopener noreferrer" className="text-primary-400 hover:underline">
                SL Mar
              </a>
            </p>
            <p>
              Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
              you may not use this software except in compliance with the License.
              You may obtain a copy of the License at:
            </p>
            <p>
              <a
                href="https://www.apache.org/licenses/LICENSE-2.0"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary-400 hover:underline break-all"
              >
                https://www.apache.org/licenses/LICENSE-2.0
              </a>
            </p>
            <p>
              Unless required by applicable law or agreed to in writing, software
              distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
              WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
              See the License for the specific language governing permissions and
              limitations under the License.
            </p>
          </div>
        </section>

        {/* Version */}
        <div className="text-center text-xs text-gray-600 pt-4">
          v0.1.0 &middot; windmar-nav/windmar
        </div>
      </div>
    </div>
  );
}
