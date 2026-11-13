"""D5 failure handling over the byte-frozen prepared R3-A framed transport.

send/recv, framing, authentication, sequence/shape/hash checks and success
records delegate to the legacy Channel. Only the failure diagnostics differ.
"""
import sys
from .communication import Channel as PreparedChannel


class Channel(PreparedChannel):
    def _record(self,operation,h,start,wire,error=None,**kwargs):
        if error is None:return super()._record(operation,h,start,wire,error=error,**kwargs)
        # The frozen send/recv except block calls _record while its original
        # exception is active. Keep it even if trace export or close then fails.
        primary=sys.exc_info()[1]
        if primary is None:return super()._record(operation,h,start,wire,error=error,**kwargs)
        self._d5_primary=primary;self.failed=True
        try:return super()._record(operation,h,start,wire,error=error,**kwargs)
        except BaseException as exc:
            if hasattr(primary,'add_note'):primary.add_note('failure trace export: '+repr(exc))
            # PreparedChannel has already appended the failure record. Its
            # error handler ignores this return value and still closes/raises.
            return None

    def _operation(self,method,*args,**kwargs):
        self._d5_primary=None
        try:return method(*args,**kwargs)
        except BaseException as secondary:
            primary=self._d5_primary
            self.failed=True
            if primary is None:
                primary=secondary
                try:self.sock.close()
                except BaseException as exc:
                    if hasattr(primary,'add_note'):primary.add_note('failed channel close: '+repr(exc))
            elif primary is not secondary:
                # The frozen error handler attempted close, which raised.
                if hasattr(primary,'add_note'):primary.add_note('failed channel close: '+repr(secondary))
            if primary is secondary:raise
            raise primary.with_traceback(primary.__traceback__) from None
        finally:self._d5_primary=None

    def send(self,h,payload=b''):
        return self._operation(super().send,h,payload)

    def recv(self,**expected):
        return self._operation(super().recv,**expected)
