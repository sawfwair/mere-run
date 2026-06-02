import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';
import { handleRequest } from './index.js';

const originalFetch = globalThis.fetch;
const docsOrigin = 'https://docs.mere.run';

afterEach(() => {
	globalThis.fetch = originalFetch;
});

function assetEnv(response = new Response('docs page', { status: 200 })) {
	const requests = [];
	return {
		env: {
			AUTH_CLIENT_ID: 'docs',
			AUTH_INTERNAL_TOKEN: 'internal-token',
			AUTH_BROKER_ORIGIN: 'https://mere.world',
			ASSETS: {
				async fetch(request) {
					requests.push(new URL(request.url).pathname);
					return response;
				}
			}
		},
		requests
	};
}

function fetchInputUrl(input) {
	return new URL(input instanceof Request ? input.url : input.toString());
}

function fetchInitHeaders(input, init) {
	return input instanceof Request ? input.headers : new Headers(init?.headers);
}

async function fetchInitJson(init) {
	return JSON.parse(String(init?.body ?? '{}'));
}

describe('docs host auth worker', () => {
	it('redirects unauthenticated requests to the Mere World broker', async () => {
		const { env } = assetEnv();
		const response = await handleRequest(new Request(`${docsOrigin}/guide/getting-started`), env);
		const location = new URL(response.headers.get('location'));

		assert.equal(response.status, 302);
		assert.equal(location.origin, 'https://mere.world');
		assert.equal(location.pathname, '/sign-in');
		assert.equal(location.searchParams.get('origin'), docsOrigin);
		assert.equal(location.searchParams.get('path'), '/guide/getting-started');
		assert.equal(location.searchParams.get('client_id'), 'docs');
	});

	it('exchanges callback codes and sets a docs app cookie', async () => {
		globalThis.fetch = async (request, init) => {
			assert.equal(fetchInputUrl(request).pathname, '/api/auth/app/exchange');
			assert.equal(fetchInitHeaders(request, init).get('authorization'), 'Bearer internal-token');
			assert.deepEqual(await fetchInitJson(init), {
				code: 'grant_123',
				audienceOrigin: docsOrigin,
				clientId: 'docs',
				userAgent: null
			});
			return Response.json({
				sessionToken: 'asess_docs',
				redirectPath: '/guide/',
				session: { expiresAt: new Date(Date.now() + 120_000).toISOString() }
			});
		};

		const { env } = assetEnv();
		const response = await handleRequest(
			new Request(`${docsOrigin}/auth/callback?code=grant_123`),
			env
		);

		assert.equal(response.status, 302);
		assert.equal(response.headers.get('location'), '/guide/');
		assert.match(response.headers.get('set-cookie') ?? '', /mere_docs_session=asess_docs/);
		assert.match(response.headers.get('set-cookie') ?? '', /HttpOnly/);
		assert.match(response.headers.get('set-cookie') ?? '', /Secure/);
	});

	it('introspects app sessions before serving VitePress assets', async () => {
		globalThis.fetch = async (request, init) => {
			assert.equal(fetchInputUrl(request).pathname, '/api/auth/session/introspect');
			assert.deepEqual(await fetchInitJson(init), {
				token: 'asess_docs',
				audienceOrigin: docsOrigin,
				clientId: 'docs'
			});
			return Response.json({
				sessionId: 'session_1',
				clientId: 'docs',
				user: { userId: 'user_1', email: 'owner@example.com' }
			});
		};

		const { env, requests } = assetEnv(new Response('docs asset', { status: 200 }));
		const response = await handleRequest(
			new Request(`${docsOrigin}/guide/getting-started`, {
				headers: { cookie: 'mere_docs_session=asess_docs' }
			}),
			env
		);

		assert.equal(response.status, 200);
		assert.equal(await response.text(), 'docs asset');
		assert.deepEqual(requests, ['/guide/getting-started.html']);
		assert.equal(response.headers.get('x-mere-docs-auth'), 'valid');
	});

	it('clears stale app session cookies before redirecting', async () => {
		globalThis.fetch = async () => Response.json(null);

		const { env, requests } = assetEnv(new Response('should not serve', { status: 200 }));
		const response = await handleRequest(
			new Request(`${docsOrigin}/`, {
				headers: { cookie: 'mere_docs_session=stale_docs_session' }
			}),
			env
		);
		const location = new URL(response.headers.get('location'));

		assert.equal(response.status, 302);
		assert.equal(location.origin, 'https://mere.world');
		assert.match(response.headers.get('set-cookie') ?? '', /mere_docs_session=;/);
		assert.match(response.headers.get('set-cookie') ?? '', /Max-Age=0/);
		assert.deepEqual(requests, []);
	});
});
