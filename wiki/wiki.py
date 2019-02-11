import asyncio
import aiohttp
import mwparserfromhell as mwp

async def get_raw_wiki(client, url):
  print('Starting {}'.format(url))
  response = await client.get(url)
  data = await response.text()
  return data

async def main():
  async with aiohttp.ClientSession() as client:
    url = "https://en.wikipedia.org/wiki/World_War_II?action=raw"
    text = await get_raw_wiki(client, url)
    print('{}: {} bytes: {}'.format(url, len(text), text))

if __name__ == "__main__":
  loop = asyncio.get_event_loop()
  loop.run_until_complete(main())
