import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Deep navy maritime colour palette
        primary: {
          50: '#eaeff8',
          100: '#c4d0eb',
          200: '#9db1de',
          300: '#7692d1',
          400: '#587bc8',
          500: '#3a5eae',
          600: '#2d4a89',
          700: '#213664',
          800: '#14223f',
          900: '#080e1a',
        },
        ocean: {
          50: '#eaeff4',
          100: '#c5d0de',
          200: '#9fb0c8',
          300: '#7991b2',
          400: '#5c7aa1',
          500: '#43638a',
          600: '#344e6c',
          700: '#25394e',
          800: '#172430',
          900: '#090f12',
        },
        maritime: {
          dark: '#060c19',
          darker: '#030710',
          light: '#0d1828',
          lighter: '#17253c',
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-maritime': 'linear-gradient(135deg, #060c19 0%, #0d1828 100%)',
        'gradient-ocean': 'linear-gradient(135deg, #25394e 0%, #43638a 100%)',
      },
      boxShadow: {
        'maritime': '0 4px 20px rgba(58, 94, 174, 0.15)',
        'ocean': '0 4px 20px rgba(67, 99, 138, 0.15)',
      },
    },
  },
  plugins: [],
}

export default config
