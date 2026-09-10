"""Local continuity contracts; HTTP and HA are stubs, not live acceptance."""
import asyncio
import unittest
import uuid
from unittest import mock

from tests.test_support import FakeConversationInput
from custom_components.hermes_conversation import conversation as conversation_module
from tests.test_api import FakeResponse, FakeSession
from tests.test_session_safety import make_agent


class ChatLog:
    def __init__(self, conversation_id):
        self.conversation_id = conversation_id

    async def async_add_delta_content_stream(self, agent_id, stream):
        async for delta in stream:
            yield delta


class ContinuityTests(unittest.IsolatedAsyncioTestCase):
    async def test_established_chat_log_not_raw_input_or_device_defines_scope(self):
        agent, transport = make_agent([FakeResponse() for _ in range(4)], enable_session_reuse=False)
        # A missing/stale incoming ID must not override HA's established chat log.
        for established, incoming in [('a', None), ('b', None), ('a', 'stale'), ('b', 'stale')]:
            result = await agent._async_handle_message(FakeConversationInput(
                'Hello', conversation_id=incoming, device_id='same-device',
                satellite_id='same-satellite'), ChatLog(established))
            self.assertEqual(result.conversation_id, established)
        ids = [call['headers']['X-Hermes-Session-Id'] for call in transport.calls]
        self.assertNotEqual(ids[0], ids[1])
        self.assertEqual(ids[0], ids[2])
        self.assertEqual(ids[1], ids[3])
        self.assertEqual(transport.calls[0]['json'], transport.calls[1]['json'])
        for sid in ids:
            self.assertEqual(uuid.UUID(sid).version, 4)
        self.assertEqual(set(agent.session_map), {'conversation:a', 'conversation:b'})

    async def test_reuse_off_expiry_zero_timeout_and_new_agent(self):
        agent, transport = make_agent([FakeResponse() for _ in range(4)],
                                      enable_session_reuse=False, session_timeout_seconds=10)
        async def turn():
            await agent._async_handle_message(FakeConversationInput('Hello'), ChatLog('same'))
        with mock.patch.object(conversation_module.time, 'time', return_value=100):
            await turn()
        # Preserve existing strict-greater-than idle timeout behavior.
        with mock.patch.object(conversation_module.time, 'time', return_value=110):
            await turn()
        with mock.patch.object(conversation_module.time, 'time', return_value=121):
            await turn()
        agent.entry.options['session_timeout_seconds'] = 0
        with mock.patch.object(conversation_module.time, 'time', return_value=10000):
            await turn()
        ids = [call['headers']['X-Hermes-Session-Id'] for call in transport.calls]
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(ids[1], ids[2])
        self.assertEqual(ids[2], ids[3])
        # Setup creates a new map; the old client diagnostic ID is not routing state.
        reloaded, reload_transport = make_agent([FakeResponse()], enable_session_reuse=False)
        reloaded.client._last_session_id = ids[-1]
        await reloaded._async_handle_message(FakeConversationInput('Hello'), ChatLog('same'))
        fresh = reload_transport.calls[0]['headers']['X-Hermes-Session-Id']
        self.assertNotIn(fresh, ids)
        self.assertEqual(uuid.UUID(fresh).version, 4)

    async def test_legacy_generated_conversation_id_can_be_continued(self):
        agent, transport = make_agent([FakeResponse() for _ in range(3)], enable_session_reuse=False)
        with mock.patch.object(conversation_module, 'async_get_chat_log', None):
            first = await agent.async_process(FakeConversationInput('Hello', conversation_id=None))
            await agent.async_process(FakeConversationInput('again', conversation_id=first.conversation_id))
            second = await agent.async_process(FakeConversationInput('Hello', conversation_id=None))
        ids = [call['headers']['X-Hermes-Session-Id'] for call in transport.calls]
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(ids[0], ids[2])
        self.assertNotEqual(first.conversation_id, second.conversation_id)

    async def test_unauthenticated_reuse_off_ignores_existing_maps(self):
        agent, transport = make_agent([FakeResponse(), FakeResponse()], key='', enable_session_reuse=False)
        agent.session_map['conversation:same'] = {'session_id': 'must-not-send', 'last_used_at': 1}
        for _ in range(2):
            await agent._async_handle_message(FakeConversationInput('Hello'), ChatLog('same'))
        self.assertTrue(all('X-Hermes-Session-Id' not in call['headers'] for call in transport.calls))
        self.assertIn({'role': 'user', 'content': 'Hello'}, transport.calls[1]['json']['messages'][:-1])

    async def test_reuse_on_retains_satellite_scope_and_device_precedence(self):
        agent, transport = make_agent([FakeResponse() for _ in range(4)])
        for conv, device in [('a', None), ('b', None), ('c', 'device'), ('d', 'device')]:
            await agent._async_handle_message(FakeConversationInput(
                'Hello', device_id=device, satellite_id='satellite'), ChatLog(conv))
        ids = [call['headers']['X-Hermes-Session-Id'] for call in transport.calls]
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[2], ids[3])
        self.assertNotEqual(ids[0], ids[2])
        self.assertEqual(set(agent.session_map), {'satellite:satellite', 'device:device'})

    async def test_stable_header_addresses_complete_stub_backend_trace(self):
        # Contract model only: emulate explicit-ID lookup ignoring body history,
        # including tool records never returned to HA. This is NOT Hermes execution.
        tool_trace = [
            {'role': 'assistant', 'tool_calls': [{'id': 'call-1', 'type': 'function',
              'function': {'name': 'read_sensor', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'private-tool-detail'},
            {'role': 'assistant', 'content': 'Visible answer'},
        ]
        class TraceTransport(FakeSession):
            def __init__(self):
                super().__init__([FakeResponse(), FakeResponse(), FakeResponse()])
                self.traces = {}
                self.loaded = []

            def post(self, url, **kwargs):
                sid = kwargs['headers']['X-Hermes-Session-Id']
                self.loaded.append(list(self.traces.get(sid, [])))
                self.traces.setdefault(sid, []).extend([kwargs['json']['messages'][-1], *tool_trace])
                return super().post(url, **kwargs)

        agent, _ = make_agent([], enable_session_reuse=False)
        transport = TraceTransport()
        agent.client._session = transport
        for text, conv in [('Hello', 'a'), ('followup', 'a'), ('Hello', 'b')]:
            await agent._async_handle_message(FakeConversationInput(text), ChatLog(conv))
        self.assertEqual(transport.loaded[1], [{'role': 'user', 'content': 'Hello'}, *tool_trace])
        self.assertEqual(transport.loaded[2], [])
        self.assertNotIn('private-tool-detail', str(transport.calls[1]['json']))

    async def test_reuse_off_concurrent_distinct_chats_keep_rotated_stream_and_json_ids(self):
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                started, release = asyncio.Event(), asyncio.Event()
                class SlowResponse(FakeResponse):
                    async def iter_any(self):
                        started.set()
                        await release.wait()
                        yield b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n'
                responses: list[FakeResponse] = [SlowResponse(headers={'X-Hermes-Session-Id': 'rotated-A'})]
                if fallback:
                    responses.append(FakeResponse(status=400))
                responses.extend([FakeResponse(headers={'X-Hermes-Session-Id': 'rotated-B'}),
                                  FakeResponse(), FakeResponse()])
                agent, transport = make_agent(responses, enable_session_reuse=False)
                async def turn(conv):
                    await agent._async_handle_message(FakeConversationInput(
                        'Hello', device_id='same'), ChatLog(conv))
                async def second():
                    await started.wait()
                    try:
                        await turn('b')
                    finally:
                        release.set()
                await asyncio.wait_for(asyncio.gather(turn('a'), second()), timeout=2)
                self.assertNotEqual(transport.calls[0]['headers']['X-Hermes-Session-Id'],
                                    transport.calls[1]['headers']['X-Hermes-Session-Id'])
                await turn('a')
                await turn('b')
                self.assertEqual(transport.calls[-2]['headers']['X-Hermes-Session-Id'], 'rotated-A')
                self.assertEqual(transport.calls[-1]['headers']['X-Hermes-Session-Id'], 'rotated-B')
                if fallback:
                    self.assertEqual(transport.calls[1]['headers'], transport.calls[2]['headers'])

    async def test_late_response_cannot_restore_invalidated_conversation(self):
        for invalidation in ('expiry', 'clear', 'rotation'):
            with self.subTest(invalidation=invalidation):
                started, release = asyncio.Event(), asyncio.Event()

                class SlowResponse(FakeResponse):
                    async def iter_any(self):
                        started.set()
                        await release.wait()
                        yield b'data: {"choices":[{"delta":{"content":"old"}}]}\n\n'

                agent, transport = make_agent([
                    SlowResponse(headers={'X-Hermes-Session-Id': 'old-rotation'}),
                    FakeResponse(headers={'X-Hermes-Session-Id': 'new-rotation'}),
                    FakeResponse(),
                ], enable_session_reuse=False)

                async def replace():
                    await started.wait()
                    try:
                        if invalidation == 'expiry':
                            agent.session_map['conversation:same']['last_used_at'] = 1
                        elif invalidation == 'clear':
                            agent.session_map.clear()
                        await agent._async_handle_message(FakeConversationInput('new'), ChatLog('same'))
                    finally:
                        release.set()

                await asyncio.wait_for(asyncio.gather(
                    agent._async_handle_message(FakeConversationInput('old'), ChatLog('same')),
                    replace()), timeout=2)
                self.assertEqual(agent.session_map['conversation:same']['session_id'], 'new-rotation')
                await agent._async_handle_message(FakeConversationInput('followup'), ChatLog('same'))
                self.assertEqual(transport.calls[-1]['headers']['X-Hermes-Session-Id'], 'new-rotation')
                initial_ids = [call['headers']['X-Hermes-Session-Id'] for call in transport.calls[:2]]
                if invalidation == 'rotation':
                    self.assertEqual(initial_ids[0], initial_ids[1])
                else:
                    self.assertNotEqual(initial_ids[0], initial_ids[1])

    async def test_overlapping_first_turns_reserve_one_conversation_id(self):
        started, release = asyncio.Event(), asyncio.Event()

        class SlowResponse(FakeResponse):
            async def iter_any(self):
                started.set()
                await release.wait()
                yield b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n'

        agent, transport = make_agent(
            [SlowResponse(), FakeResponse()], enable_session_reuse=False)

        async def second():
            await started.wait()
            try:
                await agent._async_handle_message(FakeConversationInput('two'), ChatLog('same'))
            finally:
                release.set()

        await asyncio.wait_for(asyncio.gather(
            agent._async_handle_message(FakeConversationInput('one'), ChatLog('same')),
            second()), timeout=2)
        ids = [call['headers']['X-Hermes-Session-Id'] for call in transport.calls]
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(agent.session_map['conversation:same']['session_id'], ids[0])
