import asyncio
import json
import sys
from urllib.parse import urlparse

def check_policy_update(rb_host, rb_port):
    pass

def update_policy(rb_host, rb_port):
    pass

async def tcp_echo_client(loop, mj_host, mj_port):
    reader, writer = await asyncio.open_connection(mj_host, mj_port, loop=loop)
    while True:
        data = await reader.readline()
        received = json.loads(data.decode().strip())

        print(f'in\t{received}')

        type = received['type']

        if type == 'hello':
            response = {
                'type': 'join',
                'name': 'selfplay_agent',
                'room': None
            }
        elif type == 'start_game':
            id = received['id']
            response = {'type': 'none'}
        elif type == 'end_game':
            break
        elif type == 'tsumo':
            if received['actor'] == id:
                response = {
                    'type': 'dahai',
                    'actor': id,
                    'pai': received['pai'],
                    'tsumogiri': True,
                }
            else:
                response = {'type': 'none'}
        elif type == 'error':
            break
        else:
            response = {'type': 'none'}

        send_string = json.dumps(response, separators=(',', ':')) + '\n'
        print(f'out\t{send_string}')

        writer.write(send_string.encode())
        await writer.drain()



if __name__ == '__main__':
    print(sys.argv)
    parsed = [
        urlparse(sys.argv[1]),
        urlparse(sys.argv[2])
    ]
    mj_host, mj_port = parsed[0].netloc.split(':')
    rb_host, rb_port = parsed[1].netloc.split(':')

    while True:
        if check_policy_update(rb_host, rb_port):
            update_policy(rb_host, rb_port)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            tcp_echo_client(loop, mj_host, int(mj_port))
        )
        loop.close()
