def step(self, action: int):
    if self.event_idx >= len(self.events):
        return self._observation(), 0.0, True, False, {"reason": "end_of_replay"}

    previous_mid_ticks = Decimal(str(self.book.mid_price_int() or 0))

    proposal = self.mapper.to_proposal(int(action))
    first_event = self.events[self.event_idx]
    start_ms = self._event_time(first_event)

    risk_result = self.engine.propose_and_place(
        proposal,
        start_ms,
        self.last_market_event_ms or start_ms,
    )

    fills = []
    processed_events = 0
    depth_events = 0
    trade_events = 0

    while self.event_idx < len(self.events):
        event = self.events[self.event_idx]
        event_ms = self._event_time(event)

        if processed_events > 0 and event_ms - start_ms >= self.cfg.decision_interval_ms:
            break

        self.event_idx += 1
        processed_events += 1

        if event.kind == "depth":
            self.book.apply_depth_event(event.payload)
            self.last_market_event_ms = int(event.payload.get("E", event_ms))
            depth_events += 1

        elif event.kind == "aggTrade":
            trade = AggTrade.from_binance(
                event.payload,
                self.cfg.tick_size,
                self.cfg.step_size,
            )
            self.trade_agg.add(trade)
            new_fills = self.engine.process_trade(trade)
            fills.extend(new_fills)
            self.last_market_event_ms = trade.trade_time_ms
            trade_events += 1

        else:
            raise ValueError(f"unknown event kind {event.kind}")

    current_mid_ticks = Decimal(str(self.book.mid_price_int() or previous_mid_ticks))

    reward = self.reward_model.compute(
        portfolio=self.engine.portfolio,
        fills=fills,
        previous_mid=previous_mid_ticks,
        current_mid=current_mid_ticks,
        risk_penalty=risk_result.penalty,
    )

    terminated = self.event_idx >= len(self.events)

    mid = self.engine.current_mid_decimal()
    info = {
        "risk": risk_result.reason,
        "risk_decision": str(risk_result.decision),
        "reward_breakdown": reward,
        "fills": len(fills),
        "processed_events": processed_events,
        "depth_events": depth_events,
        "trade_events": trade_events,
        "open_orders": len(self.engine.open_orders),
        "inventory": float(self.engine.portfolio.inventory),
        "cash": float(self.engine.portfolio.cash),
        "equity": float(self.engine.portfolio.equity(mid)),
    }

    return self._observation(), reward.total, terminated, False, info