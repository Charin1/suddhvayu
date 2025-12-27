/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: {
                    DEFAULT: '#2BD42E', // Primary / Green
                    soft: '#55DD58',    // Secondary / Green Soft
                },
                aqi: {
                    good: '#2BD42E',
                    moderate: '#55DD58',
                    warning: '#FFC107',
                    danger: '#FF5252',
                },
                dark: {
                    base: '#0E1117',    // BG / Main
                    card: '#161B22',    // BG / Card
                },
                text: {
                    primary: '#E6EDF3', // Text / Primary
                    muted: '#8B949E',   // Text / Muted
                },
                background: '#0E1117',
                foreground: '#E6EDF3',
                card: {
                    DEFAULT: '#161B22',
                    foreground: '#E6EDF3'
                },
                muted: {
                    DEFAULT: '#161B22',
                    foreground: '#8B949E'
                },
                border: '#30363d', // GitHub dark border approximation
            },
            fontFamily: {
                sans: ['Inter', 'SF Pro', 'Manrope', 'system-ui', 'sans-serif'],
            },
        },
    },
    plugins: [],
}
