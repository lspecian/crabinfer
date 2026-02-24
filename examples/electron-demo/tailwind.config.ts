import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./src/renderer/**/*.{html,tsx,ts}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#1a1a2e',
          light: '#232340',
          lighter: '#2d2d50',
        },
        accent: {
          DEFAULT: '#e94560',
          light: '#f06e85',
        },
        crab: {
          DEFAULT: '#ff6b35',
          light: '#ff8c5e',
        },
      },
    },
  },
  plugins: [],
}

export default config
