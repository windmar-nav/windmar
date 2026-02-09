import type { Metadata } from 'next'
import './globals.css'
import { Providers } from './providers'
import { VoyageProvider } from '@/components/VoyageContext'

export const metadata: Metadata = {
  title: 'WINDMAR - Marine Route Analysis',
  description: 'Marine route analysis for MR Product Tankers',
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
          <VoyageProvider>{children}</VoyageProvider>
        </Providers>
      </body>
    </html>
  )
}
