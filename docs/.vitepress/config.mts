import { defineConfig } from 'vitepress'
import { existsSync } from 'node:fs'

function resolveBase(): string {
  const explicit = process.env.DOCS_BASE?.trim()
  if (explicit) {
    return explicit.startsWith('/') ? explicit : `/${explicit}`
  }

  if (existsSync(new URL('../public/CNAME', import.meta.url))) {
    return '/'
  }

  const repository = process.env.GITHUB_REPOSITORY?.split('/')[1]
  if (repository) {
    return repository.endsWith('.github.io') ? '/' : `/${repository}/`
  }

  return '/'
}

export default defineConfig({
  title: 'mere.run',
  description: 'Local-first inference on Apple Silicon with a public Swift package and CLI.',
  base: resolveBase(),
  cleanUrls: true,
  lastUpdated: true,
  srcExclude: ['README.md', 'SUMMARY.md'],
  themeConfig: {
    siteTitle: 'mere.run Docs',
    search: {
      provider: 'local'
    },
    nav: [
      { text: 'Getting Started', link: '/getting-started' },
      { text: 'CLI', link: '/cli' },
      { text: 'Cookbooks', link: '/cookbooks' },
      { text: 'Runtime', link: '/runtime/image' },
      { text: 'Internals', link: '/internals/cli-and-runtime' }
    ],
    sidebar: [
      {
        text: 'Introduction',
        items: [
          { text: 'Docs Home', link: '/' },
          { text: 'Getting Started', link: '/getting-started' },
          { text: 'CLI Reference', link: '/cli' },
          { text: 'Cookbooks', link: '/cookbooks' },
          { text: 'Configuration', link: '/configuration' },
          { text: 'Model Sources', link: '/model-sources' }
        ]
      },
      {
        text: 'Repository Guides',
        items: [
          { text: 'Repository Tour', link: '/repository-tour' },
          { text: 'Development Workflow', link: '/development-workflow' },
          { text: 'Testing Guide', link: '/testing' },
          { text: 'Architecture Reading Map', link: '/architecture' }
        ]
      },
      {
        text: 'Runtime Families',
        items: [
          { text: 'Image Runtime', link: '/runtime/image' },
          { text: 'Text Runtime', link: '/runtime/text' },
          { text: 'Speech Runtime', link: '/runtime/speech' },
          { text: 'Vision Runtime', link: '/runtime/vision' },
          { text: 'Music Runtime', link: '/runtime/music' },
          { text: 'Video Runtime', link: '/runtime/video' },
          { text: 'Model Management', link: '/runtime/model-management' },
          { text: 'Local API Server', link: '/runtime/api-server' }
        ]
      },
      {
        text: 'Internals',
        items: [
          { text: 'CLI and Runtime Internals', link: '/internals/cli-and-runtime' },
          { text: 'Source Layout Reference', link: '/internals/source-layout' }
        ]
      }
    ],
    outline: {
      level: [2, 3]
    },
    docFooter: {
      prev: 'Previous',
      next: 'Next'
    },
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © mere.run contributors'
    }
  }
})
