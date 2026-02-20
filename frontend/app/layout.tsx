import type { Metadata } from 'next'
import './globals.css'
import { Providers } from './providers'
import { VoyageProvider } from '@/components/VoyageContext'
import { DemoFooter } from '@/components/DemoFooter'
import { AuthGate } from '@/components/AuthGate'

export const metadata: Metadata = {
  title: 'WINDMAR - Weather Routing & Performance Analytics',
  description: 'Weather routing and performance analytics for merchant ships',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        />
        <link
          rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css"
        />
      </head>
      <body>
        <Providers>
          <AuthGate>
            <VoyageProvider>{children}</VoyageProvider>
            <DemoFooter />
          </AuthGate>
        </Providers>
      </body>
    </html>
  )
}
