/// <reference path="./vue-shim.d.ts" />

import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import MereLayout from './MereLayout.vue'
import {
  defaultMereDocsThemeConfig,
  provideMereDocsThemeConfig,
  resolveMereProductDocsKeyColor,
  type MereAtlasPlane,
  type MereDocsKeyColorInput,
  type MereDocsThemeUserConfig,
} from './config.js'
import './mere-theme.css'

export * from './config.js'

export interface MereProductDocsThemeOptions {
  productName: string
  productDomain: string
  docsUrl: string
  productHref?: string
  keyColor?: MereDocsKeyColorInput
  corePrefix?: string
  coreSuffix?: string
  guideHref?: string
  architectureHref?: string
  operationsHref?: string
  workflowsHref?: string
  runtimeHref?: string
  pluginsHref?: string
  referenceHref?: string
  cliHref?: string
  /** Override the generic four-plane atlas with product-specific copy. */
  planes?: MereAtlasPlane[]
}

export function createMereDocsTheme(config: MereDocsThemeUserConfig = {}): Theme {
  return {
    extends: DefaultTheme,
    Layout: MereLayout,
    enhanceApp(ctx) {
      DefaultTheme.enhanceApp?.(ctx)
      provideMereDocsThemeConfig(ctx.app, config)
    },
  }
}

export function createMereProductDocsTheme(options: MereProductDocsThemeOptions): Theme {
  const productHref = options.productHref ?? options.docsUrl
  const coreParts = options.productDomain.split('.')
  const corePrefix = options.corePrefix ?? coreParts.at(0) ?? 'mere'
  const coreSuffix = options.coreSuffix ?? (coreParts.slice(1).join('.') || 'docs')
  const guideHref = options.guideHref ?? '/'
  const architectureHref = options.architectureHref ?? guideHref
  const operationsHref = options.operationsHref ?? guideHref
  const workflowsHref = options.workflowsHref ?? guideHref
  const runtimeHref = options.runtimeHref ?? architectureHref
  const referenceHref = options.referenceHref ?? guideHref
  const pluginsHref = options.pluginsHref ?? referenceHref
  const cliHref = options.cliHref ?? referenceHref
  const planes: MereAtlasPlane[] = options.planes ?? [
    {
      name: `${options.productName} CLI`,
      signal: 'Complete command tree, setup paths, and practical cookbooks',
      href: referenceHref,
      accent: 'green',
      items: ['Commands', 'Cookbooks', 'Setup'],
    },
    {
      name: 'Portable workflows',
      signal: 'Immutable bundles across local, SSH, and relay executors',
      href: workflowsHref,
      accent: 'blue',
      items: ['Graphs', 'Executors', 'Runs'],
    },
    {
      name: 'Runtime families',
      signal: 'Image, text, audio, vision, video, 3D, and persistent worlds',
      href: runtimeHref,
      accent: 'plum',
      items: ['Multimodal', 'Resident', 'Local'],
    },
    {
      name: 'Companion plugins',
      signal: 'Verified external tools and typed graph providers',
      href: pluginsHref,
      accent: 'copper',
      items: ['Catalog', 'Doctor', 'Providers'],
    },
  ]

  return createMereDocsTheme({
    keyColor: resolveMereProductDocsKeyColor(options.productName, options.productDomain, options.keyColor),
    atlas: {
      eyebrowLeft: options.docsUrl.replace(/^https?:\/\//, '').replace(/\/$/, ''),
      eyebrowRight: 'docs online',
      corePrefix,
      coreSuffix,
      planes,
    },
    docsNetworkLinks: [
      { text: 'Mere atlas', href: 'https://mere-docs.mere.world/' },
      { text: options.productName, href: productHref },
      { text: 'This docs site', href: options.docsUrl },
      { text: 'Mere World auth', href: 'https://docs.mere.world/' },
    ],
    sectionSignals: [
      {
        match: ['/guide/', '/getting-started', '/index'],
        label: `${options.productName} guide`,
        detail: 'User workflows, setup paths, and day-to-day product use',
        primaryHref: guideHref,
        primaryText: 'Docs home',
        secondaryHref: guideHref,
        secondaryText: 'Get started',
      },
      {
        match: ['/architecture/', '/concepts/', '/internals/', '/platform/'],
        label: 'Architecture',
        detail: 'System boundaries, runtime shape, and implementation contracts',
        primaryHref: architectureHref,
        primaryText: 'Architecture',
        secondaryHref: referenceHref,
        secondaryText: 'Reference',
      },
      {
        match: ['/operations/', '/ops/', '/development/', '/contributing/'],
        label: 'Operations',
        detail: 'Deploy, verify, troubleshoot, and release this surface',
        primaryHref: operationsHref,
        primaryText: 'Operations',
        secondaryHref: architectureHref,
        secondaryText: 'Development',
      },
      {
        match: ['/api/', '/reference/', '/cli/', '/commands/', '/runtime/'],
        label: 'Reference',
        detail: 'APIs, commands, runtime contracts, and generated surfaces',
        primaryHref: referenceHref,
        primaryText: 'Reference',
        secondaryHref: cliHref,
        secondaryText: 'CLI',
      },
      ...defaultMereDocsThemeConfig.sectionSignals,
    ],
    defaultSectionSignal: {
      label: options.productName,
      detail: `Product docs for ${options.productDomain}`,
      primaryHref: guideHref,
      primaryText: 'Docs home',
      secondaryHref: 'https://mere-docs.mere.world/',
      secondaryText: 'Mere atlas',
    },
  })
}

export default createMereDocsTheme()
