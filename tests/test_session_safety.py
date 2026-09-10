"""Session safety regressions using the real API client under upstream HA stubs."""
import asyncio
import unittest
import uuid
from tests.test_support import FakeConfigEntry, FakeConversationInput, FakeHass
from tests.test_api import FakeSession, FakeResponse
from custom_components.hermes_conversation.api import HermesApiClient
from custom_components.hermes_conversation.conversation import HermesConversationAgent


def make_agent(responses, key='test-only', **options):
    transport = FakeSession(responses)
    client = HermesApiClient(transport, 'localhost', 1234, key, use_ssl=False)
    agent = HermesConversationAgent(FakeHass(), FakeConfigEntry(
        data={'api_key': key}, options={'prompt': '', 'expose_device_context': False, **options}), client, {})
    return agent, transport


class SessionSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_identical_authenticated_openings_get_random_explicit_ids(self):
        agent, transport = make_agent([FakeResponse(), FakeResponse()])
        for conv in ('a', 'b'):
            await agent.async_process(FakeConversationInput('Hello', conversation_id=conv))
        ids = [c['headers'].get('X-Hermes-Session-Id') for c in transport.calls]
        self.assertTrue(all(ids), 'First authenticated POST must have a session ID')
        self.assertNotEqual(*ids)
        for sid in ids:
            self.assertEqual(uuid.UUID(sid).version, 4)
        self.assertEqual(transport.calls[0]['json'], transport.calls[1]['json'])

    async def test_authenticated_reuse_off_continues_same_conversation(self):
        agent, transport = make_agent([FakeResponse(), FakeResponse()], enable_session_reuse=False)
        for text in ('one', 'two'):
            await agent.async_process(FakeConversationInput(text, conversation_id='a'))
        ids = [c['headers']['X-Hermes-Session-Id'] for c in transport.calls]
        self.assertEqual(ids[0], ids[1], 'Follow-ups must address the same backend trace')
        self.assertEqual(uuid.UUID(ids[0]).version, 4)
        self.assertEqual(agent.session_map['conversation:a']['session_id'], ids[0])

    async def test_unauthenticated_requests_remain_headerless(self):
        agent, transport = make_agent([FakeResponse(), FakeResponse()], key='')
        for text in ('one', 'two'):
            await agent.async_process(FakeConversationInput(text, conversation_id='a'))
        self.assertTrue(all('X-Hermes-Session-Id' not in c['headers'] for c in transport.calls))
        self.assertEqual(agent.session_map, {})

    async def test_device_reuse_expiry_zero_timeout_and_reload(self):
        agent, transport = make_agent([FakeResponse() for _ in range(5)])
        async def turn(conv):
            await agent.async_process(FakeConversationInput('Hello', conversation_id=conv, device_id='voice'))
        await turn('a')
        await turn('b')
        first = transport.calls[0]['headers']['X-Hermes-Session-Id']
        self.assertEqual(transport.calls[1]['headers']['X-Hermes-Session-Id'], first)
        agent.session_map['device:voice']['last_used_at'] = 1
        await turn('c')
        fresh = transport.calls[2]['headers']['X-Hermes-Session-Id']
        self.assertNotEqual(fresh, first)
        agent.entry.options['session_timeout_seconds'] = 0
        agent.session_map['device:voice']['last_used_at'] = 1
        await turn('d')
        self.assertEqual(transport.calls[3]['headers']['X-Hermes-Session-Id'], fresh)
        agent.session_map.clear()
        await turn('e')
        self.assertNotEqual(transport.calls[4]['headers']['X-Hermes-Session-Id'], fresh)

    async def test_chat_log_child_task_retains_request_metadata(self):
        class ChatLog:
            conversation_id = 'modern'
            async def async_add_delta_content_stream(self, agent_id, stream):
                async def consume():
                    return [delta async for delta in stream]
                for delta in await asyncio.create_task(consume()):
                    yield delta
        agent, _ = make_agent([FakeResponse(headers={'X-Hermes-Session-Id': 'rotated-modern'})])
        await agent._async_handle_message(FakeConversationInput('Hello'), ChatLog())
        self.assertEqual(agent.session_map['conversation:modern']['session_id'], 'rotated-modern')

    async def test_concurrent_stream_and_json_do_not_crosswire_rotated_ids(self):
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                first_started, second_finished = asyncio.Event(), asyncio.Event()
                class SlowStream(FakeResponse):
                    async def iter_any(self):
                        first_started.set()
                        await second_finished.wait()
                        yield b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n'
                responses = [SlowStream(headers={'X-Hermes-Session-Id': 'rotated-A'})]
                if fallback:
                    responses += [FakeResponse(status=400), FakeResponse(
                        headers={'X-Hermes-Session-Id': 'rotated-B'},
                        json_data={'choices': [{'message': {'content': 'B'}}]})]
                else:
                    responses += [FakeResponse(headers={'X-Hermes-Session-Id': 'rotated-B'})]
                agent, _ = make_agent(responses)
                async def second():
                    await first_started.wait()
                    await agent.async_process(FakeConversationInput('B', conversation_id='b'))
                    second_finished.set()
                await asyncio.wait_for(asyncio.gather(
                    agent.async_process(FakeConversationInput('A', conversation_id='a')),
                    second()), timeout=2)
                self.assertEqual(agent.session_map['conversation:a']['session_id'], 'rotated-A')
                self.assertEqual(agent.session_map['conversation:b']['session_id'], 'rotated-B')
