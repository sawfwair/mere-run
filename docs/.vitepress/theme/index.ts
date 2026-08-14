import { createMereProductDocsTheme } from './mere/index.js'

export default createMereProductDocsTheme({
  productName: 'mere.run',
  productDomain: 'docs.mere.run',
  docsUrl: 'https://docs.mere.run/',
  productHref: 'https://mere.run',
  hero: 'run-proof',
  corePrefix: 'Eight runtime families. One local CLI. No cloud in the loop.',
  guideHref: '/getting-started',
  architectureHref: '/architecture',
  operationsHref: '/development-workflow',
  workflowsHref: '/workflows',
  runtimeHref: '/runtime/image',
  pluginsHref: '/plugins',
  referenceHref: '/cli',
  cliHref: '/cli',
  planes: [
    {
      name: 'Make something',
      signal: 'Image, video, music, speech, vision, 3D, and worlds you can walk around in',
      href: '/runtime/image',
      accent: 'green',
      items: ['Eight families', 'Native MLX', 'On your disk'],
    },
    {
      name: 'Drive it from the CLI',
      signal: 'Every command and flag, plus offline cookbooks that work with the network off',
      href: '/cli',
      accent: 'blue',
      items: ['Commands', 'Cookbooks', 'Setup'],
    },
    {
      name: 'Run it anywhere',
      signal: 'Describe a job once, then run it here, over SSH, or across a fleet — same bytes out',
      href: '/workflows',
      accent: 'plum',
      items: ['Graphs', 'Executors', 'Runs'],
    },
    {
      name: 'Extend it safely',
      signal: 'Typed graph providers that add nodes without entering the mere.run process',
      href: '/plugins',
      accent: 'copper',
      items: ['Catalog', 'Doctor', 'Providers'],
    },
  ]
})
