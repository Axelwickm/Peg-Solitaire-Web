const CACHE_NAME = 'peg-solitaire-static-v1';
const ASSETS = [
	'/',
	'/index.html',
	'/board.webp',
	'/ball.webp',
	'/dist/main.js',
];

self.addEventListener('install', event => {
	event.waitUntil(
		caches
			.open(CACHE_NAME)
			.then(cache => cache.addAll(ASSETS))
			.then(() => self.skipWaiting()),
	);
});

self.addEventListener('activate', event => {
	event.waitUntil(
		caches.keys().then(keys =>
			Promise.all(
				keys.map(key => {
					if (key !== CACHE_NAME) {
						return caches.delete(key);
					}
					return Promise.resolve();
				}),
			),
		).then(() => self.clients.claim()),
	);
});

self.addEventListener('fetch', event => {
	if (event.request.method !== 'GET') {
		return;
	}
	event.respondWith(
		caches.match(event.request).then(cached => {
			if (cached) {
				return cached;
			}
			return fetch(event.request)
				.then(response => {
					if (!response || response.status !== 200 || response.type !== 'basic') {
						return response;
					}
					const responseClone = response.clone();
					caches.open(CACHE_NAME).then(cache => {
						cache.put(event.request, responseClone);
					});
					return response;
				})
				.catch(() => cached || fetch(event.request));
		}),
	);
});
