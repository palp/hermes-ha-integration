"""Deterministic voice-origin lifecycle contracts (HA/HTTP stubs, not audio)."""
import asyncio
import unittest
from unittest import mock

from tests.test_support import FakeConversationInput
from custom_components.hermes_conversation import conversation as module
from tests.test_api import FakeResponse
from tests.test_conversation_continuity import ChatLog
from tests.test_session_safety import make_agent


class VoiceLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_eviction_bounds_all_scopes_and_removes_history(self):
        for reuse in (False, True):
            for origin in ('device_id', 'satellite_id', None):
                with self.subTest(reuse=reuse, origin=origin):
                    agent, transport = make_agent([FakeResponse() for _ in range(6)],
                                                  enable_session_reuse=reuse)
                    async def turn(conv):
                        fields = {origin: conv} if origin else {}
                        await agent._async_handle_message(FakeConversationInput('hello', **fields), ChatLog(conv))
                    with mock.patch.object(module, '_MAX_CACHED_CONVERSATIONS', 2):
                        await turn('a')
                        await turn('b')
                        await turn('a')  # a is most recently used
                        await turn('c')
                        self.assertEqual(len(agent.session_map), 2)
                        self.assertNotIn('conversation:b', agent.session_map)
                        self.assertNotIn('b', agent._history)
                        await turn('b')
                        self.assertNotEqual(transport.calls[1]['headers']['X-Hermes-Session-Id'],
                                            transport.calls[-1]['headers']['X-Hermes-Session-Id'])
                        if not reuse:
                            self.assertEqual(len(transport.calls[-1]['json']['messages']),
                                             len(transport.calls[1]['json']['messages']))

    async def test_busy_cache_pins_sessions_then_prunes_after_cancellation(self):
        started, release = asyncio.Event(), asyncio.Event()
        class SlowResponse(FakeResponse):
            async def iter_any(self):
                started.set()
                await release.wait()
                yield b'data: {"choices":[{"delta":{"content":"old"}}]}\n\n'
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                started.clear()
                agent, transport = make_agent([SlowResponse(), FakeResponse(), FakeResponse()],
                                              enable_session_reuse=reuse)
                async def turn(conv):
                    await agent._async_handle_message(FakeConversationInput('hello', device_id=conv), ChatLog(conv))
                with mock.patch.object(module, '_MAX_CACHED_CONVERSATIONS', 1):
                    task = asyncio.create_task(turn('a'))
                    await asyncio.wait_for(started.wait(), 2)
                    try:
                        await turn('b')
                        await turn('a')
                        self.assertEqual(transport.calls[0]['headers']['X-Hermes-Session-Id'],
                                         transport.calls[2]['headers']['X-Hermes-Session-Id'])
                    finally:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    self.assertLessEqual(len(agent.session_map), 1)
                    self.assertLessEqual(len(agent._history), 1)


    async def test_expired_active_response_does_not_restore_old_text_history(self):
        started, release = asyncio.Event(), asyncio.Event()
        class SlowResponse(FakeResponse):
            async def iter_any(self):
                started.set()
                await release.wait()
                yield b'data: {"choices":[{"delta":{"content":"old answer"}}]}\n\n'
        agent, transport = make_agent([SlowResponse(), FakeResponse(), FakeResponse()],
                                      enable_session_reuse=False)
        async def turn(text):
            await agent._async_handle_message(FakeConversationInput(text, satellite_id='voice'), ChatLog('same'))
        async def replace():
            await started.wait()
            try:
                agent.session_map['conversation:same']['last_used_at'] = 1
                await turn('new')
            finally:
                release.set()
        await asyncio.wait_for(asyncio.gather(turn('old'), replace()), 2)
        await turn('followup')
        self.assertNotIn('old', [message['content'] for message in transport.calls[-1]['json']['messages']])

    async def test_late_voice_response_cannot_overwrite_recreated_record_with_same_id(self):
        started, release = asyncio.Event(), asyncio.Event()
        class SlowResponse(FakeResponse):
            async def iter_any(self):
                started.set()
                await release.wait()
                yield b'data: {"choices":[{"delta":{"content":"old"}}]}\n\n'
        replacement = FakeResponse()
        agent, transport = make_agent([SlowResponse(headers={'X-Hermes-Session-Id': 'stale'}), replacement])
        async def turn():
            await agent._async_handle_message(FakeConversationInput('hello', device_id='speaker'), ChatLog('voice'))
        async def replace():
            await started.wait()
            try:
                original = transport.calls[0]['headers']['X-Hermes-Session-Id']
                agent.session_map.clear()
                replacement.headers['X-Hermes-Session-Id'] = original
                await turn()
            finally:
                release.set()
        await asyncio.wait_for(asyncio.gather(turn(), replace()), 2)
        self.assertEqual(agent.session_map['device:speaker']['session_id'],
                         transport.calls[0]['headers']['X-Hermes-Session-Id'])

    async def test_voice_overlap_reserves_id_and_rejects_late_rotation(self):
        for origin in ({'device_id': 'speaker'}, {'satellite_id': 'assist.speaker'}, {}):
            for invalidation in ('rotation', 'expiry', 'clear'):
                with self.subTest(origin=origin, invalidation=invalidation):
                    started, release = asyncio.Event(), asyncio.Event()

                    class SlowResponse(FakeResponse):
                        async def iter_any(self):
                            started.set()
                            await release.wait()
                            yield b'data: {"choices":[{"delta":{"content":"old"}}]}\n\n'

                    agent, transport = make_agent([
                        SlowResponse(headers={'X-Hermes-Session-Id': 'old'}),
                        FakeResponse(headers={'X-Hermes-Session-Id': 'replacement'}),
                        FakeResponse(),
                    ])
                    key = agent._build_session_key(FakeConversationInput('one', **origin), 'same')

                    async def turn(text):
                        return await agent._async_handle_message(
                            FakeConversationInput(text, **origin), ChatLog('same'))

                    async def replace():
                        await started.wait()
                        try:
                            if invalidation == 'expiry':
                                agent.session_map[key]['last_used_at'] = 1
                            elif invalidation == 'clear':
                                agent.session_map.clear()
                            await turn('two')
                        finally:
                            release.set()

                    await asyncio.wait_for(asyncio.gather(turn('one'), replace()), 2)
                    ids = [call['headers']['X-Hermes-Session-Id'] for call in transport.calls]
                    if invalidation == 'rotation':
                        self.assertEqual(ids[0], ids[1])
                    else:
                        self.assertNotEqual(ids[0], ids[1])
                    await turn('followup')
                    self.assertEqual(transport.calls[-1]['headers']['X-Hermes-Session-Id'], 'replacement')
