/**
 * Markdown/Shiki configuration for Mere docs hosts.
 *
 * Import from `@mere/docs-theme/markdown` inside `.vitepress/config.ts`.
 * Deliberately plain JavaScript with no relative imports: the VitePress
 * config loads in Node, and Node refuses to type-strip `.ts` files that
 * resolve under `node_modules`, which is exactly where this package lives
 * for every consumer repo.
 *
 * ```ts
 * import { defineConfig } from 'vitepress'
 * import { mereMarkdown } from '@mere/docs-theme/markdown'
 *
 * export default defineConfig({
 *   markdown: mereMarkdown(),
 * })
 * ```
 */

const ink = {
  fg: '#DCDAD1', // base text — warm off-white
  muted: '#8E978B', // punctuation, operators
  faint: '#7C877A', // comments
  periwinkle: '#AFBBF4', // keywords, control flow (docs key)
  sage: '#A6D9AE', // strings (world/finance key)
  gold: '#E5C97F', // functions, headings (today key)
  copper: '#E7A688', // numbers, constants (gives key)
  teal: '#8FD4CD', // types, classes, tags (network key)
  sky: '#A5CDEE', // properties, attributes (email key)
  plum: '#D8ABD3', // regex, special (media key)
  red: '#EC9E92', // deletions, errors
}

/**
 * "Mere Code" — the shared syntax palette for every Mere docs host.
 *
 * One dark theme used in BOTH site appearances: code renders as a dark
 * instrument panel on warm paper in light mode, and carved into the canvas
 * in dark mode. Token hues are drawn from the Mere product key colors
 * (periwinkle, sage, gold, copper, teal, sky, plum) and tuned to stay
 * readable (≥ ~4.5:1) on the oklch(14–22%) block backgrounds.
 */
export const mereCodeTheme = {
  name: 'mere-code',
  type: 'dark',
  colors: {
    'editor.background': '#141A17',
    'editor.foreground': ink.fg,
    'terminal.background': '#141A17',
    'terminal.foreground': ink.fg,
  },
  tokenColors: [
    { scope: ['text', 'source'], settings: { foreground: ink.fg } },
    {
      scope: ['comment', 'punctuation.definition.comment'],
      settings: { foreground: ink.faint, fontStyle: 'italic' },
    },
    {
      scope: [
        'punctuation',
        'meta.brace',
        'punctuation.definition.tag',
        'punctuation.separator',
        'punctuation.terminator',
        'meta.punctuation',
      ],
      settings: { foreground: ink.muted },
    },
    {
      scope: ['keyword.operator', 'storage.type.function.arrow'],
      settings: { foreground: ink.muted },
    },
    {
      scope: [
        'keyword',
        'keyword.control',
        'storage',
        'storage.type',
        'storage.modifier',
        'keyword.other.important',
        'variable.language.this',
        'variable.language.self',
        'variable.language.super',
      ],
      settings: { foreground: ink.periwinkle },
    },
    {
      scope: [
        'string',
        'string.quoted',
        'string.template',
        'punctuation.definition.string',
        'markup.inline.raw.string',
      ],
      settings: { foreground: ink.sage },
    },
    {
      scope: [
        'entity.name.function',
        'support.function',
        'meta.function-call.generic',
        'variable.function',
        'meta.require',
      ],
      settings: { foreground: ink.gold },
    },
    {
      scope: [
        'constant.numeric',
        'constant.language',
        'constant.character',
        'constant.other',
        'keyword.other.unit',
        'variable.other.constant',
        'support.constant',
      ],
      settings: { foreground: ink.copper },
    },
    {
      scope: [
        'entity.name.type',
        'entity.name.class',
        'entity.name.namespace',
        'entity.other.inherited-class',
        'support.type',
        'support.class',
        'entity.name.tag',
        'meta.type.name',
      ],
      settings: { foreground: ink.teal },
    },
    {
      scope: [
        'variable.other.property',
        'variable.other.object.property',
        'support.type.property-name',
        'support.type.property-name.json',
        'meta.object-literal.key',
        'entity.other.attribute-name',
        'entity.name.tag.yaml',
      ],
      settings: { foreground: ink.sky },
    },
    {
      scope: ['variable', 'variable.parameter', 'variable.other', 'meta.definition.variable'],
      settings: { foreground: ink.fg },
    },
    {
      scope: [
        'string.regexp',
        'constant.character.escape',
        'punctuation.definition.template-expression',
        'keyword.other.template',
        'entity.name.label',
        'meta.decorator',
        'punctuation.decorator',
      ],
      settings: { foreground: ink.plum },
    },
    {
      scope: ['invalid', 'invalid.illegal'],
      settings: { foreground: ink.red },
    },
    // ── Shell — the most common language in operator docs
    {
      scope: ['source.shell variable.other', 'variable.other.normal.shell', 'variable.other.special.shell'],
      settings: { foreground: ink.sky },
    },
    {
      scope: ['support.function.builtin.shell', 'entity.name.command.shell'],
      settings: { foreground: ink.gold },
    },
    // ── CSS
    {
      scope: ['entity.other.attribute-name.class.css', 'entity.other.attribute-name.id.css'],
      settings: { foreground: ink.gold },
    },
    {
      scope: ['entity.other.attribute-name.pseudo-class.css', 'entity.other.attribute-name.pseudo-element.css'],
      settings: { foreground: ink.plum },
    },
    {
      scope: ['support.type.vendored.property-name.css'],
      settings: { foreground: ink.sky },
    },
    // ── Markdown
    {
      scope: ['markup.heading', 'entity.name.section', 'punctuation.definition.heading'],
      settings: { foreground: ink.gold },
    },
    { scope: 'markup.bold', settings: { fontStyle: 'bold' } },
    { scope: 'markup.italic', settings: { fontStyle: 'italic' } },
    {
      scope: ['markup.underline.link', 'string.other.link', 'constant.other.reference.link'],
      settings: { foreground: ink.teal },
    },
    {
      scope: ['markup.quote'],
      settings: { foreground: ink.faint, fontStyle: 'italic' },
    },
    {
      scope: ['markup.inserted', 'punctuation.definition.inserted'],
      settings: { foreground: ink.sage },
    },
    {
      scope: ['markup.deleted', 'punctuation.definition.deleted'],
      settings: { foreground: ink.red },
    },
    {
      scope: ['markup.changed', 'punctuation.definition.changed'],
      settings: { foreground: ink.gold },
    },
  ],
}

/**
 * Returns the `markdown` options every Mere docs host should use.
 * Spread your own additions after it if a host needs more:
 *
 * ```ts
 * markdown: { ...mereMarkdown(), math: true }
 * ```
 *
 * @param {{ lineNumbers?: boolean }} [options]
 */
export function mereMarkdown(options = {}) {
  return {
    // One dark theme for both appearances — code is always an instrument
    // panel. The block background itself comes from --vp-code-block-bg in
    // mere-theme.css so each mode can tune how the panel sits on the page.
    theme: /** @type {never} */ (mereCodeTheme),
    lineNumbers: options.lineNumbers ?? false,
  }
}
