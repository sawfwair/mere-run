import { inject, type App, type InjectionKey } from 'vue'

export type MereAtlasAccent = 'green' | 'blue' | 'plum' | 'copper'

export interface MereDocsKeyColor {
  light: string
  lightHover: string
  lightContrast: string
  dark: string
  darkHover: string
  darkContrast: string
}

function defineMereDocsKeyColor(
  light: string,
  lightHover: string,
  dark: string,
  darkHover: string,
  lightContrast = '#FCFCFB',
  darkContrast = '#171714',
): MereDocsKeyColor {
  return {
    light,
    lightHover,
    lightContrast,
    dark,
    darkHover,
    darkContrast,
  }
}

export const mereDocsKeyColors = {
  docs: defineMereDocsKeyColor('#3B4EA0', '#2F3E82', '#AEBBFF', '#CAD2FF'),
  world: defineMereDocsKeyColor('#0E7A4E', '#0B623F', '#65D6A4', '#96E9C3'),
  business: defineMereDocsKeyColor('#24566E', '#1B4559', '#75C0DA', '#A0DAEA'),
  agent: defineMereDocsKeyColor('#6E3878', '#582C61', '#D69CDF', '#E8C2EE'),
  email: defineMereDocsKeyColor('#2D6C9B', '#22567D', '#8BC9F0', '#B6DDF6'),
  today: defineMereDocsKeyColor('#9C6F1A', '#7C5814', '#E8C469', '#F2D991', '#171714', '#171714'),
  gives: defineMereDocsKeyColor('#A84E2C', '#873D22', '#EA9A78', '#F2BCA7'),
  network: defineMereDocsKeyColor('#1C7373', '#165C5C', '#73D6D1', '#A1E8E3'),
  dynasite: defineMereDocsKeyColor('#27708B', '#1D5A70', '#81CADF', '#AEE0EA'),
  video: defineMereDocsKeyColor('#9A4C37', '#7B3A29', '#E69C86', '#F0BEAD'),
  ink: defineMereDocsKeyColor('#1A1A17', '#11110F', '#D5D2C8', '#E8E5DC'),
  works: defineMereDocsKeyColor('#415B91', '#334874', '#9DB9F1', '#C0D1F7'),
  finance: defineMereDocsKeyColor('#1F7044', '#185937', '#7AD39C', '#A6E7BD'),
  fit: defineMereDocsKeyColor('#527A2A', '#416221', '#A9D477', '#C7E79F', '#171714', '#171714'),
  earth: defineMereDocsKeyColor('#66712B', '#535C22', '#C3D277', '#DBE79F', '#171714', '#171714'),
  im: defineMereDocsKeyColor('#23668F', '#1A5273', '#82C6ED', '#AFDCF6'),
  media: defineMereDocsKeyColor('#873E78', '#6C315F', '#DB9BD0', '#EDBFE7'),
  news: defineMereDocsKeyColor('#965D24', '#784A1C', '#E2B16A', '#EECF94'),
  projects: defineMereDocsKeyColor('#7A5B2A', '#614820', '#D8B876', '#E7D09B'),
  zone: defineMereDocsKeyColor('#733E93', '#5B3175', '#C9A0EC', '#DDC0F4'),
  deliver: defineMereDocsKeyColor('#A1661B', '#805014', '#E6BE6D', '#F0D493', '#171714', '#171714'),
  run: defineMereDocsKeyColor('#3159A8', '#274687', '#93B5F4', '#B9CDF8'),
} satisfies Record<string, MereDocsKeyColor>

export type MereDocsKeyColorName = keyof typeof mereDocsKeyColors
export type MereDocsKeyColorInput = MereDocsKeyColorName | Partial<MereDocsKeyColor>

export interface MereAtlasPlane {
  name: string
  signal: string
  href: string
  accent: MereAtlasAccent
  items: string[]
}

export interface MereDocsLink {
  text: string
  href: string
}

export interface MereDocSectionSignal {
  match: string[]
  label: string
  detail: string
  primaryHref: string
  primaryText: string
  secondaryHref: string
  secondaryText: string
}

export interface MereDocsThemeConfig {
  keyColor: MereDocsKeyColor
  atlas: {
    eyebrowLeft: string
    eyebrowRight: string
    corePrefix: string
    coreSuffix: string
    planes: MereAtlasPlane[]
  }
  docsNetworkLabel: string
  docsNetworkLinks: MereDocsLink[]
  sectionSignals: MereDocSectionSignal[]
  defaultSectionSignal: Omit<MereDocSectionSignal, 'match'>
}

export type MereDocsThemeUserConfig = Partial<{
  keyColor: MereDocsKeyColorInput
  atlas: Partial<MereDocsThemeConfig['atlas']>
  docsNetworkLabel: string
  docsNetworkLinks: MereDocsLink[]
  sectionSignals: MereDocSectionSignal[]
  defaultSectionSignal: Partial<MereDocsThemeConfig['defaultSectionSignal']>
}>

export const defaultMereDocsThemeConfig: MereDocsThemeConfig = {
  keyColor: mereDocsKeyColors.docs,
  atlas: {
    eyebrowLeft: 'mere-docs.mere.world',
    eyebrowRight: 'atlas online',
    corePrefix: 'mere',
    coreSuffix: 'docs',
    planes: [
      {
        name: 'Product docs',
        signal: 'Customer workflows, product fit, setup paths',
        href: '/product/',
        accent: 'green',
        items: ['Business', 'World', 'Agent'],
      },
      {
        name: 'Dedicated hosts',
        signal: 'Worker-first docs for each app surface',
        href: '/reference/documentation-coverage-matrix',
        accent: 'blue',
        items: ['docs.mere.*', 'merekit docs', 'mere.run'],
      },
      {
        name: 'Ecosystem atlas',
        signal: 'Architecture, operations, release gates',
        href: '/ecosystem/overview',
        accent: 'plum',
        items: ['Topology', 'Contracts', 'Runbooks'],
      },
      {
        name: 'Local runtime',
        signal: 'Private data, local AI, command planes',
        href: '/runtime/mere-run-and-native',
        accent: 'copper',
        items: ['mere.run', 'MereKit', 'Local plane'],
      },
    ],
  },
  docsNetworkLabel: 'Docs network',
  docsNetworkLinks: [
    { text: 'Mere atlas', href: '/' },
    { text: 'Products', href: '/product/products/' },
    { text: 'Coverage matrix', href: '/reference/documentation-coverage-matrix' },
    { text: 'Mere World docs', href: 'https://docs.mere.world/' },
  ],
  sectionSignals: [
    {
      match: ['/product/'],
      label: 'Product docs',
      detail: 'Customer-facing workflows and app boundaries',
      primaryHref: '/product/products/',
      primaryText: 'Product directory',
      secondaryHref: '/product/what-can-i-do',
      secondaryText: 'Choose by outcome',
    },
    {
      match: ['/ecosystem/', '/apps/'],
      label: 'Ecosystem atlas',
      detail: 'Repository ownership, app surfaces, and traversal order',
      primaryHref: '/ecosystem/repository-catalog',
      primaryText: 'Repository catalog',
      secondaryHref: '/apps/product-apps',
      secondaryText: 'App map',
    },
    {
      match: ['/architecture/'],
      label: 'Architecture',
      detail: 'Runtime contracts, integration boundaries, and portfolio shape',
      primaryHref: '/architecture/system-atlas',
      primaryText: 'System atlas',
      secondaryHref: '/architecture/auth-and-runtime-contracts',
      secondaryText: 'Auth contracts',
    },
    {
      match: ['/operations/'],
      label: 'Operations',
      detail: 'Verification, release, local dev, and incident signals',
      primaryHref: '/operations/ecosystem-verification',
      primaryText: 'Verification',
      secondaryHref: '/operations/cross-app-release-sequence',
      secondaryText: 'Release sequence',
    },
    {
      match: ['/reference/'],
      label: 'Reference',
      detail: 'Inventories, matrices, ports, naming, and dependency truth',
      primaryHref: '/reference/documentation-coverage-matrix',
      primaryText: 'Docs matrix',
      secondaryHref: '/reference/ecosystem-inventory',
      secondaryText: 'Inventory',
    },
    {
      match: ['/runtime/', '/tools/', '/extensions/'],
      label: 'Command and runtime',
      detail: 'Local AI, command planes, plugins, and agent-ready artifacts',
      primaryHref: '/runtime/mere-run-and-native',
      primaryText: 'mere.run',
      secondaryHref: '/tools/command-plane',
      secondaryText: 'Command plane',
    },
  ],
  defaultSectionSignal: {
    label: 'Mere docs',
    detail: 'Product guide, ecosystem atlas, and operator reference',
    primaryHref: '/product/',
    primaryText: 'Product docs',
    secondaryHref: '/ecosystem/overview',
    secondaryText: 'Ecosystem docs',
  },
}

const mereDocsThemeConfigKey: InjectionKey<MereDocsThemeConfig> = Symbol('mereDocsThemeConfig')

export function resolveMereDocsKeyColor(input: MereDocsKeyColorInput = 'docs'): MereDocsKeyColor {
  if (typeof input === 'string') {
    return mereDocsKeyColors[input] ?? mereDocsKeyColors.docs
  }

  return {
    ...mereDocsKeyColors.docs,
    ...input,
  }
}

const mereProductKeyColorAliases: Array<[string, MereDocsKeyColorName]> = [
  ['mere-docs', 'docs'],
  ['mere docs', 'docs'],
  ['mer-finance', 'finance'],
  ['mere.fi', 'finance'],
  ['mere-cli', 'run'],
  ['cli-docs', 'run'],
  ['merekit-cli', 'run'],
  ['merekit link', 'works'],
  ['merekit-link', 'works'],
  ['sites.merekit', 'dynasite'],
  ['merekit sites', 'dynasite'],
  ['meresmb', 'projects'],
  ['agent', 'agent'],
  ['business', 'business'],
  ['deliver', 'deliver'],
  ['dynasite', 'dynasite'],
  ['earth', 'earth'],
  ['email', 'email'],
  ['finance', 'finance'],
  ['fit', 'fit'],
  ['gives', 'gives'],
  ['ink', 'ink'],
  ['media', 'media'],
  ['network', 'network'],
  ['news', 'news'],
  ['projects', 'projects'],
  ['run', 'run'],
  ['today', 'today'],
  ['video', 'video'],
  ['works', 'works'],
  ['world', 'world'],
  ['zone', 'zone'],
  ['im', 'im'],
]

function normalizeMereKeyMatch(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function hasMereKeyMatch(term: string, needle: string): boolean {
  const normalizedNeedle = normalizeMereKeyMatch(needle)
  return (
    term === normalizedNeedle ||
    term.startsWith(`${normalizedNeedle}-`) ||
    term.endsWith(`-${normalizedNeedle}`) ||
    term.includes(`-${normalizedNeedle}-`)
  )
}

export function resolveMereProductDocsKeyColor(
  productName: string,
  productDomain: string,
  keyColor?: MereDocsKeyColorInput,
): MereDocsKeyColor {
  if (keyColor) {
    return resolveMereDocsKeyColor(keyColor)
  }

  const searchTerms = Array.from(
    new Set(
      [productName, productDomain, productDomain.replace(/^docs\./, '')]
        .map(normalizeMereKeyMatch)
        .filter(Boolean),
    ),
  )
  const match = mereProductKeyColorAliases.find(([needle]) =>
    searchTerms.some((term) => hasMereKeyMatch(term, needle)),
  )
  return resolveMereDocsKeyColor(match?.[1] ?? 'docs')
}

export function defineMereDocsThemeConfig(config: MereDocsThemeUserConfig): MereDocsThemeUserConfig {
  return config
}

export function resolveMereDocsThemeConfig(config: MereDocsThemeUserConfig = {}): MereDocsThemeConfig {
  return {
    ...defaultMereDocsThemeConfig,
    ...config,
    keyColor: resolveMereDocsKeyColor(config.keyColor ?? defaultMereDocsThemeConfig.keyColor),
    atlas: {
      ...defaultMereDocsThemeConfig.atlas,
      ...config.atlas,
      planes: config.atlas?.planes ?? defaultMereDocsThemeConfig.atlas.planes,
    },
    docsNetworkLinks: config.docsNetworkLinks ?? defaultMereDocsThemeConfig.docsNetworkLinks,
    sectionSignals: config.sectionSignals ?? defaultMereDocsThemeConfig.sectionSignals,
    defaultSectionSignal: {
      ...defaultMereDocsThemeConfig.defaultSectionSignal,
      ...config.defaultSectionSignal,
    },
  }
}

export function provideMereDocsThemeConfig(app: App, config?: MereDocsThemeUserConfig): void {
  app.provide(mereDocsThemeConfigKey, resolveMereDocsThemeConfig(config))
}

export function useMereDocsThemeConfig(): MereDocsThemeConfig {
  return inject(mereDocsThemeConfigKey, defaultMereDocsThemeConfig)
}
