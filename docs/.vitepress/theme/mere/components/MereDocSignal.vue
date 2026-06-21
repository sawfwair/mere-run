<script setup lang="ts">
import { computed } from 'vue'
import { useData, useRoute } from 'vitepress'
import { useMereDocsThemeConfig } from '../config'

const route = useRoute()
const { page } = useData()
const themeConfig = useMereDocsThemeConfig()

const section = computed(() => {
  const path = route.path
  return themeConfig.sectionSignals.find((signal) => signal.match.some((prefix) => path.startsWith(prefix)))
    ?? themeConfig.defaultSectionSignal
})

const lastUpdated = computed(() => {
  const timestamp = page.value.lastUpdated
  if (!timestamp) {
    return null
  }
  return new Intl.DateTimeFormat('en', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(new Date(timestamp))
})
</script>

<template>
  <aside class="mere-doc-signal" aria-label="Current docs section">
    <div>
      <span class="mere-doc-signal-label">{{ section.label }}</span>
      <p>{{ section.detail }}</p>
    </div>
    <nav class="mere-doc-signal-links" aria-label="Related docs">
      <a :href="section.primaryHref">{{ section.primaryText }}</a>
      <a :href="section.secondaryHref">{{ section.secondaryText }}</a>
      <span v-if="lastUpdated">Updated {{ lastUpdated }}</span>
    </nav>
  </aside>
</template>
