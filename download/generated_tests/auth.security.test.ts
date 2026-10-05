import { readFileSync } from 'fs';
import { describe, it, expect } from '@jest/core';

describe('security guardrails', () => {
  it('has no hardcoded secrets in src/', () => {
    const src = readFileSync('src/auth.ts', 'utf-8');
    expect(src).not.toMatch(/(?:JWT_SECRET|API_KEY|PASSWORD)\s*=\s*["'][^"']{8,}["']/);
  });

  it('does not call eval() with user input', () => {
    const src = readFileSync('src/auth.ts', 'utf-8');
    expect(src).not.toMatch(/eval\s*\(\s*req/);
  });

  it('uses HTTPS for external calls', () => {
    const src = readFileSync('src/auth.ts', 'utf-8');
    expect(src).not.toMatch(/http:\/\/(?!localhost|127\.0\.0\.1)/);
  });
});
