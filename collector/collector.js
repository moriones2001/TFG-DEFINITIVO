// collector.js
// Purpose: Connects to Twitch IRC, listens for chat messages, and pushes each message as JSON to Redis list 'incoming'.

require('dotenv').config();
const tmi = require('tmi.js');
const { createClient } = require('redis');

// Inicializar cliente Redis usando la URL de entorno
const redisClient = createClient({ url: process.env.REDIS_URL });
redisClient.connect().catch(err => {
  console.error('Redis connection error:', err);
  process.exit(1);
});

// Configuración de Twitch IRC con tmi.js
const client = new tmi.Client({
  options: { debug: false },
  identity: {
    username: process.env.TWITCH_BOT_USERNAME,
    password: process.env.TWITCH_OAUTH_TOKEN
  },
  channels: process.env.TWITCH_CHANNELS.split(',')
});

// Conectar al IRC de Twitch
client.connect().catch(err => {
  console.error('Twitch IRC connection error:', err);
  process.exit(1);
});

// Escuchar mensajes
client.on('message', async (channel, tags, message, self) => {
  if (self) return; // Ignorar mensajes del bot

  const msg = {
    channel: channel.replace('#', ''),            // nombre del canal sin '#'
    user: tags['display-name'] || tags.username,  // nombre de usuario
    text: message,                                 // contenido del mensaje
    timestamp: new Date().toISOString()            // marca de tiempo ISO
  };

  try {
    // Empujar al comienzo de la lista 'incoming'
    await redisClient.lPush('incoming', JSON.stringify(msg));
    console.log('Pushed message to Redis:', msg);
  } catch (err) {
    console.error('Error pushing to Redis:', err);
  }
});
