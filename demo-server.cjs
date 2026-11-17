const http = require('http');
const fs = require('fs');
const path = require('path');

const root = __dirname;
const port = process.env.PORT || 4173;
const types = {
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.html': 'text/html',
};
// Only these files are public (keeps server code and repo files private)
const allowed = new Set(['index.html', 'styles.css', 'script.js']);

http
  .createServer((req, res) => {
    let requested;
    try {
      const pathname = new URL(req.url, 'http://localhost').pathname;
      requested = pathname === '/' ? 'index.html' : decodeURIComponent(pathname).replace(/^\/+/, '');
    } catch {
      res.writeHead(400);
      res.end('Bad request');
      return;
    }

    const file = path.resolve(root, requested);
    const inRoot = file.startsWith(root + path.sep);
    if (!inRoot || !allowed.has(path.relative(root, file)) || !fs.existsSync(file)) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }

    res.writeHead(200, {
      'Content-Type': `${types[path.extname(file)] || 'application/octet-stream'}; charset=utf-8`,
    });
    fs.createReadStream(file).pipe(res);
  })
  .listen(port, () => console.log(`Signvoice AI demo: http://localhost:${port}`));
