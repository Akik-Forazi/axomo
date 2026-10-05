import { authMiddleware } from '../src/auth';
import { describe, it, expect, jest } from '@jest/core';

describe('authMiddleware boundary coverage', () => {
  const mkReq = (token) => ({ headers: { authorization: token }, query: {} });

  // Generated from edge-case analysis on diff:
    // token = null
  // token = "" (empty string)
  // token = "Bearer " (no payload)
  // token = 'A' (length < 10)

  it('rejects null token (was NPE)', () => {
    const res = { status: jest.fn().returnThis(), send: jest.fn() };
    authMiddleware(mkReq(null), res, () => {});
    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('rejects empty string token (was bypass)', () => {
    const res = { status: jest.fn().returnThis(), send: jest.fn() };
    authMiddleware(mkReq(''), res, () => {});
    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('rejects "Bearer " with no payload (was infinite loop)', () => {
    const res = { status: jest.fn().returnThis(), send: jest.fn() };
    authMiddleware(mkReq('Bearer '), res, () => {});
    expect(res.status).toHaveBeenCalledWith(401);
  });
});
