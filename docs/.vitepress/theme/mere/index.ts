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
  referenceHref?: string
  cliHref?: string
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
  const referenceHref = options.referenceHref ?? guideHref
  const cliHref = options.cliHref ?? referenceHref
  const planes: MereAtlasPlane[] = [
    {
      name: `${options.productName} guide`,
      signal: 'Setup paths, product workflows, and common tasks',
      href: guideHref,
      accent: 'green',
      items: ['Guide', 'Workflows', 'Setup'],
    },
    {
      name: 'Architecture',
      signal: 'Boundaries, data flow, and integration contracts',
      href: architectureHref,
      accent: 'blue',
      items: ['Contracts', 'Runtime', 'Data'],
    },
    {
      name: 'Operations',
      signal: 'Deploy, verify, release, and troubleshoot',
      href: operationsHref,
      accent: 'plum',
      items: ['Deploy', 'Testing', 'Runbooks'],
    },
    {
      name: 'Mere atlas',
      signal: 'Cross-product docs and ecosystem inventory',
      href: 'https://mere-docs.mere.world/',
      accent: 'copper',
      items: ['Products', 'Matrix', 'Reference'],
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
