export interface MereCodeThemeTokenColor {
  scope: string | string[]
  settings: {
    foreground?: string
    fontStyle?: string
  }
}

export interface MereCodeTheme {
  name: string
  type: 'dark'
  colors: Record<string, string>
  tokenColors: MereCodeThemeTokenColor[]
}

/** "Mere Code" — the shared syntax palette for every Mere docs host. */
export declare const mereCodeTheme: MereCodeTheme

export interface MereMarkdownOptions {
  /** Show line numbers on all code blocks. Default false. */
  lineNumbers?: boolean
}

/**
 * Returns the `markdown` options every Mere docs host should use.
 * Spread your own additions after it if a host needs more:
 * `markdown: { ...mereMarkdown(), math: true }`
 */
export declare function mereMarkdown(options?: MereMarkdownOptions): {
  theme: never
  lineNumbers: boolean
}
