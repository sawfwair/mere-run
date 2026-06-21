import { inject, type App, type InjectionKey } from 'vue'

export type MereAtlasAccent = 'green' | 'blue' | 'plum' | 'copper'

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
  atlas: Partial<MereDocsThemeConfig['atlas']>
  docsNetworkLabel: string
  docsNetworkLinks: MereDocsLink[]
  sectionSignals: MereDocSectionSignal[]
  defaultSectionSignal: Partial<MereDocsThemeConfig['defaultSectionSignal']>
}>

export const defaultMereDocsThemeConfig: MereDocsThemeConfig = {
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

export function defineMereDocsThemeConfig(config: MereDocsThemeUserConfig): MereDocsThemeUserConfig {
  return config
}

export function resolveMereDocsThemeConfig(config: MereDocsThemeUserConfig = {}): MereDocsThemeConfig {
  return {
    ...defaultMereDocsThemeConfig,
    ...config,
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
