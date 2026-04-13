FROM node:20-slim

WORKDIR /app

COPY package.json tsconfig.json ./
RUN npm install --production=false

COPY src/ ./src/
RUN npm run build

EXPOSE 3000

CMD ["node", "dist/index.js"]
