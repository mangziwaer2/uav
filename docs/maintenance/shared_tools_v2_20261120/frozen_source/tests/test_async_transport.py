import threading

import pytest

from uav_em.async_transport import ThreadAdvisor


def test_incomplete_callback_poll_is_nonblocking_and_cancellation_does_not_free_worker():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    def callback(public):
        started.set()
        try:
            if not release.wait(2.):
                raise RuntimeError('test callback release timed out')
            public['value'].append('worker change')
            return {'action': 'keep'}
        finally:
            finished.set()
    advisor = ThreadAdvisor(callback)
    payload = dict(value=['original'])
    try:
        token = advisor.submit(payload, 0.)
        assert started.wait(1.)
        assert advisor.poll(1.) == [] and advisor.busy()
        assert not advisor.cancel(token) and advisor.busy()
        with pytest.raises(ValueError, match='busy'):
            advisor.submit({}, 1.)
        release.set()
        assert finished.wait(1.)
        advisor.current['future'].result(timeout=1.)
        reply = advisor.poll(2.)[0]
        assert reply['token'] == token and reply['actual_model_call'] is None
        assert reply['started_wall_seconds'] <= reply['received_wall_seconds']
        assert reply['visible_at_physical_seconds'] == 2.
        assert payload == dict(value=['original'])
        assert not advisor.busy() and advisor.poll(3.) == []
    finally:
        release.set()
        advisor.close()


def test_worker_exception_is_a_reply_without_claiming_model_execution():
    def callback(_):
        raise RuntimeError('synthetic network failure')
    advisor = ThreadAdvisor(callback)
    try:
        advisor.submit({}, 0.)
        advisor.current['future'].result(timeout=1.)
        reply = advisor.poll(1.)[0]
        assert reply['raw'] is None and 'synthetic network failure' in reply['error']
        assert reply['actual_model_call'] is None
    finally:
        advisor.close()
    with pytest.raises(ValueError, match='closed'):
        advisor.submit({}, 2.)
