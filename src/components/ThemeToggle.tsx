import { useState, useEffect } from 'react'

const THEME_KEY = 'mnist-nn-theme'

type Theme = 'light' | 'dark'

function getInitialTheme(): Theme {
  const saved = localStorage.getItem(THEME_KEY)
  if (saved === 'dark' || saved === 'light') return saved
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  const toggle = () => setTheme(t => (t === 'light' ? 'dark' : 'light'))

  return (
    <button className="theme-toggle" onClick={toggle} title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}>
      <span className="icon">{theme === 'light' ? '☀️' : '🌙'}</span>
      {theme === 'light' ? 'Light' : 'Dark'}
    </button>
  )
}
