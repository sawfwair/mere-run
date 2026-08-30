import { defineConfig } from 'vitepress'
import { existsSync } from 'node:fs'
import { mereMarkdown } from './theme/mere/markdown'

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
  description: 'Local multimodal inference, portable workflow graphs, and headless executors from one public Swift CLI.',
  base: resolveBase(),
  cleanUrls: true,
  lastUpdated: true,
  srcExclude: [
    'README.md',
    'macos-studio-roadmap.md',
    'macos-studio-capability-review.md',
    'falcon-perception-disparity-report.md',
    'architecture/vfx-geometry-model-report.md',
    'benchmarks/vfx-geometry-apple-silicon.md'
  ],
  markdown: mereMarkdown(),
  themeConfig: {
    siteTitle: 'mere.run / docs',
    search: {
      provider: 'local'
    },
    nav: [
      { text: 'Create', link: '/runtime/image' },
      { text: 'Understand', link: '/runtime/vision' },
      { text: 'Automate', link: '/workflows' },
      { text: 'Serve', link: '/runtime/api-server' },
      { text: 'Reference', link: '/cli' }
    ],
    sidebar: [
      {
        text: 'Introduction',
        items: [
          { text: 'Docs home', link: '/' },
          { text: 'Getting started', link: '/getting-started' },
          { text: 'macOS deep links', link: '/macos-deep-links' },
          { text: 'Linux quickstart', link: '/linux-quickstart' },
          { text: 'CLI reference', link: '/cli' },
          { text: 'Offline cookbooks', link: '/cookbooks' },
        ]
      },
      {
        text: 'Workflows and operations',
        items: [
          { text: 'Portable workflows', link: '/workflows' },
          { text: 'Graph Studio', link: '/graph/studio' },
          { text: 'Configuration', link: '/configuration' },
          { text: 'Quality gate', link: '/gate' },
          { text: 'Model sources', link: '/model-sources' },
          { text: 'Companion plugins', link: '/plugins' },
          { text: 'Raycast example', link: '/raycast' }
        ]
      },
      {
        text: 'Benchmarks',
        items: [
          { text: 'Benchmarking overview', link: '/benchmarking' },
          { text: 'Fused model evaluation', link: '/benchmark-fused' },
          { text: 'Fused reference runs', link: '/benchmarks/fused-reference-runs' },
          { text: 'External evaluation packs', link: '/evaluation-packs' }
        ]
      },
      {
        text: 'Repository guides',
        items: [
          { text: 'Repository tour', link: '/repository-tour' },
          { text: 'Development workflow', link: '/development-workflow' },
          { text: 'Testing guide', link: '/testing' },
          { text: 'Documentation style', link: '/documentation-style' },
          { text: 'Documentation style audit', link: '/documentation-style-audit' },
          { text: 'Architecture reading map', link: '/architecture' },
          { text: 'mlx-swift fork policy', link: '/mlx-swift-fork' }
        ]
      },
      {
        text: 'Runtime families',
        items: [
          { text: 'Image runtime', link: '/runtime/image' },
          { text: 'Text runtime', link: '/runtime/text' },
          { text: 'Speech runtime', link: '/runtime/speech' },
          { text: 'Vision runtime', link: '/runtime/vision' },
          { text: 'Geospatial runtime', link: '/runtime/geo' },
          { text: 'Audio enhancement', link: '/runtime/audio' },
          { text: 'Music runtime', link: '/runtime/music' },
          { text: 'ACE-Step validation', link: '/runtime/acestep-validation' },
          { text: 'SFX runtime', link: '/runtime/sfx' },
          { text: 'Video runtime', link: '/runtime/video' },
          { text: 'Persistent worlds', link: '/runtime/world' },
          { text: 'Model management', link: '/runtime/model-management' },
          { text: 'Local API server', link: '/runtime/api-server' }
        ]
      },
      {
        text: 'Internals',
        items: [
          { text: 'CLI and runtime internals', link: '/internals/cli-and-runtime' },
          { text: 'Structured runs, preflights, and declarative actions', link: '/internals/structured-runs-preflight-actions' },
          { text: 'Source layout reference', link: '/internals/source-layout' },
          { text: 'Guarded acceleration audit', link: '/internals/guarded-acceleration' },
          { text: 'DiT forward-pass performance', link: '/internals/dit-performance' }
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
      copyright: 'Copyright © 2026 Sawfwair Inc. and mere.run contributors'
    }
  }
})
