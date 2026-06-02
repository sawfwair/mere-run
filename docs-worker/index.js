const DEFAULT_AUTH_BROKER_ORIGIN = 'https://mere.world';
const DEFAULT_LOCAL_AUTH_BROKER_ORIGIN = 'http://127.0.0.1:4312';
const DEFAULT_AUTH_COOKIE_NAME = 'mere_docs_session';
const DEFAULT_CLIENT_ID = 'docs';
const DEFAULT_SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7;

function normalizeOrigin(value) {
	try {
		return new URL(value).origin;
	} catch {
		return null;
	}
}

function isLocalDevelopmentHostname(hostname) {
	return (
		hostname === '127.0.0.1' ||
		hostname === 'localhost' ||
		hostname === '0.0.0.0' ||
		hostname === '[::1]' ||
		hostname === '::1'
	);
}

async function readBindingValue(binding) {
	if (!binding) return null;
	if (typeof binding === 'string') return binding.trim() || null;
	if (typeof binding.get === 'function') {
		try {
			const value = String(await binding.get()).trim();
			return value || null;
		} catch {
			return null;
		}
	}
	return null;
}

function clientId(env) {
	return env.AUTH_CLIENT_ID?.trim() || DEFAULT_CLIENT_ID;
}

function resolveBrokerOrigin(env, requestOrigin) {
	const configured = normalizeOrigin(env.AUTH_BROKER_ORIGIN ?? '');
	if (configured) return configured;

	try {
		return isLocalDevelopmentHostname(new URL(requestOrigin).hostname)
			? DEFAULT_LOCAL_AUTH_BROKER_ORIGIN
			: DEFAULT_AUTH_BROKER_ORIGIN;
	} catch {
		return DEFAULT_AUTH_BROKER_ORIGIN;
	}
}

function normalizeCookieClientId(value) {
	return value.trim().toLowerCase().replace(/[^a-z0-9_-]/g, '_');
}

function authCookieName(env, requestOrigin) {
	const configured = (env.AUTH_COOKIE_NAME ?? DEFAULT_AUTH_COOKIE_NAME).trim() || DEFAULT_AUTH_COOKIE_NAME;
	if (configured !== DEFAULT_AUTH_COOKIE_NAME) return configured;

	try {
		if (!isLocalDevelopmentHostname(new URL(requestOrigin).hostname)) return configured;
	} catch {
		return configured;
	}

	return `${DEFAULT_AUTH_COOKIE_NAME}_${normalizeCookieClientId(clientId(env))}`;
}

function isSecureRequest(request) {
	const url = new URL(request.url);
	return url.protocol === 'https:' || request.headers.get('x-forwarded-proto') === 'https';
}

function appSessionTokenFromRequest(request, cookieName) {
	const authorization = request.headers.get('authorization');
	if (authorization?.startsWith('Bearer ')) {
		const token = authorization.slice('Bearer '.length).trim();
		if (token) return token;
	}

	const cookieHeader = request.headers.get('cookie');
	if (!cookieHeader) return null;

	const tokens = [];
	for (const cookie of cookieHeader.split(';')) {
		const [rawName, ...rawValueParts] = cookie.trim().split('=');
		if (rawName !== cookieName) continue;
		const rawValue = rawValueParts.join('=');
		if (rawValue) tokens.push(decodeURIComponent(rawValue));
	}
	return tokens.at(-1) ?? null;
}

function normalizeRedirectPath(value, fallback = '/') {
	const raw = value?.trim();
	if (!raw) return fallback;

	const hasControlCharacter = (path) => {
		for (let index = 0; index < path.length; index += 1) {
			const code = path.charCodeAt(index);
			if (code <= 0x1f || code === 0x7f) return true;
		}
		return false;
	};

	const safePath = (path) => {
		if (!path.startsWith('/') || path.startsWith('//') || path.startsWith('/\\')) return null;
		return hasControlCharacter(path) ? null : path;
	};

	const directPath = safePath(raw);
	if (directPath) return directPath;

	try {
		const url = new URL(raw);
		return safePath(`${url.pathname}${url.search}${url.hash}`) ?? fallback;
	} catch {
		return fallback;
	}
}

function buildBrokerAuthUrl({ brokerOrigin, mode, targetOrigin, redirectPath, clientId }) {
	const url = new URL(mode === 'sign-up' ? '/sign-up' : '/sign-in', brokerOrigin);
	url.searchParams.set('origin', targetOrigin);
	url.searchParams.set('path', normalizeRedirectPath(redirectPath));
	url.searchParams.set('client_id', clientId);
	return url.toString();
}

function redirectResponse(location, headers = undefined) {
	const nextHeaders = new Headers(headers);
	nextHeaders.set('location', location);
	return new Response(null, { status: 302, headers: nextHeaders });
}

function jsonResponse(payload, status = 200) {
	return new Response(JSON.stringify(payload), {
		status,
		headers: {
			'cache-control': 'no-store',
			'content-type': 'application/json; charset=utf-8'
		}
	});
}

function serializeCookie({ name, value, secure, maxAge }) {
	return [
		`${name}=${encodeURIComponent(value)}`,
		'Path=/',
		'HttpOnly',
		'SameSite=Lax',
		secure ? 'Secure' : '',
		`Max-Age=${String(Math.max(0, Math.floor(maxAge)))}`
	]
		.filter(Boolean)
		.join('; ');
}

function clearCookie(name, secure = false) {
	return [
		`${name}=`,
		'Path=/',
		'HttpOnly',
		'SameSite=Lax',
		secure ? 'Secure' : '',
		'Max-Age=0',
		'Expires=Thu, 01 Jan 1970 00:00:00 GMT'
	]
		.filter(Boolean)
		.join('; ');
}

async function brokerJsonRequest(path, { brokerOrigin, internalToken, body }) {
	const response = await fetch(new URL(path, brokerOrigin), {
		method: 'POST',
		headers: {
			authorization: `Bearer ${internalToken}`,
			'content-type': 'application/json'
		},
		body: JSON.stringify(body)
	});

	const payload = await response.json().catch(() => null);
	if (!response.ok || !payload) {
		throw new Error(payload?.error ?? payload?.message ?? `Broker request failed (${response.status}).`);
	}
	return payload;
}

async function authInternalToken(env) {
	return (
		(await readBindingValue(env.AUTH_INTERNAL_TOKEN)) ??
		(await readBindingValue(env.INTERNAL_SERVICE_TOKEN))
	);
}

async function exchangeBrokerCode({ brokerOrigin, internalToken, code, audienceOrigin, userAgent, clientId }) {
	return brokerJsonRequest('/api/auth/app/exchange', {
		brokerOrigin,
		internalToken,
		body: { code, audienceOrigin, clientId, userAgent: userAgent ?? null }
	});
}

async function introspectAppSession({ brokerOrigin, internalToken, sessionToken, audienceOrigin, clientId }) {
	return brokerJsonRequest('/api/auth/session/introspect', {
		brokerOrigin,
		internalToken,
		body: { token: sessionToken, audienceOrigin, clientId }
	});
}

async function revokeAppSession({ brokerOrigin, internalToken, sessionToken, audienceOrigin, clientId }) {
	await brokerJsonRequest('/api/auth/session/logout', {
		brokerOrigin,
		internalToken,
		body: { token: sessionToken, audienceOrigin, clientId }
	});
}

function signInRedirect(request, env, mode = 'sign-in', headers = undefined) {
	const url = new URL(request.url);
	const redirectUrl =
		url.pathname === '/sign-in' || url.pathname === '/sign-up'
			? url.searchParams.get('redirect_url')
			: `${url.pathname}${url.search}`;
	return redirectResponse(
		buildBrokerAuthUrl({
			brokerOrigin: resolveBrokerOrigin(env, url.origin),
			mode,
			targetOrigin: url.origin,
			redirectPath: normalizeRedirectPath(redirectUrl),
			clientId: clientId(env)
		}),
		headers
	);
}

async function handleAuthCallback(request, env) {
	const url = new URL(request.url);
	const code = url.searchParams.get('code')?.trim();
	if (!code) return signInRedirect(request, env);

	const internalToken = await authInternalToken(env);
	if (!internalToken) return jsonResponse({ error: 'AUTH_INTERNAL_TOKEN is not configured.' }, 500);

	let exchange;
	try {
		exchange = await exchangeBrokerCode({
			brokerOrigin: resolveBrokerOrigin(env, url.origin),
			internalToken,
			code,
			audienceOrigin: url.origin,
			userAgent: request.headers.get('user-agent'),
			clientId: clientId(env)
		});
	} catch (error) {
		console.error('Docs auth callback exchange failed.', {
			audienceOrigin: url.origin,
			message: error instanceof Error ? error.message : String(error)
		});
		return redirectResponse('/sign-in?auth_error=callback_failed');
	}

	const expiresAt = new Date(exchange.session?.expiresAt ?? '').getTime();
	const maxAge = Number.isFinite(expiresAt)
		? Math.max(60, Math.floor((expiresAt - Date.now()) / 1000))
		: DEFAULT_SESSION_MAX_AGE_SECONDS;

	return redirectResponse(exchange.redirectPath || '/', {
		'set-cookie': serializeCookie({
			name: authCookieName(env, url.origin),
			value: exchange.sessionToken,
			secure: url.protocol === 'https:',
			maxAge
		})
	});
}

async function handleSignOut(request, env) {
	const url = new URL(request.url);
	const cookieName = authCookieName(env, url.origin);
	const sessionToken = appSessionTokenFromRequest(request, cookieName);
	const internalToken = await authInternalToken(env);

	if (sessionToken && internalToken) {
		await revokeAppSession({
			brokerOrigin: resolveBrokerOrigin(env, url.origin),
			internalToken,
			sessionToken,
			audienceOrigin: url.origin,
			clientId: clientId(env)
		}).catch(() => null);
	}

	const signOutUrl = new URL('/sign-out', resolveBrokerOrigin(env, url.origin));
	signOutUrl.searchParams.set('origin', url.origin);
	signOutUrl.searchParams.set('path', '/');
	return redirectResponse(signOutUrl.toString(), {
		'set-cookie': clearCookie(cookieName, isSecureRequest(request))
	});
}

function hasPathExtension(pathname) {
	const lastSegment = pathname.split('/').pop() ?? '';
	return lastSegment.includes('.');
}

function candidateAssetPaths(pathname) {
	if (pathname === '/') return ['/'];
	if (hasPathExtension(pathname)) return [pathname];
	if (pathname.endsWith('/')) return [pathname, `${pathname}index.html`];
	return [`${pathname}.html`, `${pathname}/index.html`, pathname];
}

async function fetchAsset(request, env) {
	if (!env.ASSETS?.fetch) return jsonResponse({ error: 'ASSETS binding is not configured.' }, 500);

	const requestUrl = new URL(request.url);
	for (const pathname of candidateAssetPaths(requestUrl.pathname)) {
		const assetUrl = new URL(request.url);
		assetUrl.pathname = pathname;
		const response = await env.ASSETS.fetch(new Request(assetUrl, request));
		if (response.status !== 404) return response;
	}

	return env.ASSETS.fetch(request);
}

async function requireSession(request, env) {
	const url = new URL(request.url);
	const cookieName = authCookieName(env, url.origin);
	const sessionToken = appSessionTokenFromRequest(request, cookieName);
	if (!sessionToken) return { cookieName, session: null, shouldClearCookie: false };

	const internalToken = await authInternalToken(env);
	if (!internalToken) return { cookieName, session: null, shouldClearCookie: false };

	const session = await introspectAppSession({
		brokerOrigin: resolveBrokerOrigin(env, url.origin),
		internalToken,
		sessionToken,
		audienceOrigin: url.origin,
		clientId: clientId(env)
	}).catch(() => null);

	return { cookieName, session, shouldClearCookie: !session?.user };
}

function withAuthDebugHeader(response, state) {
	const headers = new Headers(response.headers);
	headers.set('x-mere-docs-auth', state);
	return new Response(response.body, {
		status: response.status,
		statusText: response.statusText,
		headers
	});
}

export async function handleRequest(request, env) {
	const url = new URL(request.url);

	if (url.pathname === '/api/auth/ok') {
		return jsonResponse({
			ok: true,
			app: env.APP_LABEL ?? 'mere-docs-host',
			clientId: clientId(env),
			brokerOrigin: resolveBrokerOrigin(env, url.origin)
		});
	}

	if (url.pathname === '/sign-in') return signInRedirect(request, env, 'sign-in');
	if (url.pathname === '/sign-up') return signInRedirect(request, env, 'sign-up');
	if (url.pathname === '/auth/callback') return handleAuthCallback(request, env);
	if (url.pathname === '/sign-out') return handleSignOut(request, env);

	const sessionResult = await requireSession(request, env);
	if (!sessionResult.session?.user) {
		const authState = sessionResult.shouldClearCookie ? 'stale' : 'missing';
		return signInRedirect(
			request,
			env,
			'sign-in',
			sessionResult.shouldClearCookie
				? {
						'x-mere-docs-auth': authState,
						'set-cookie': clearCookie(sessionResult.cookieName, isSecureRequest(request))
					}
				: { 'x-mere-docs-auth': authState }
		);
	}

	return withAuthDebugHeader(await fetchAsset(request, env), 'valid');
}

export default {
	fetch: handleRequest
};
