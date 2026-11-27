import React from 'react';
import { Text, View } from 'react-native';

import { useAuth } from '../lib/auth';
import type { TKey } from '../lib/i18n';
import { colors } from '../theme';
import { Chip, Field, H2, P } from './ui';

/** The same ten items as the IFLS5 asset score the risk model (ifls5-lr-2) was trained on (backend schemas.HOUSEHOLD_ASSETS). */
export const HOUSEHOLD_ASSETS = ['own_house', 'electricity', 'fridge', 'tv', 'gas_cooking', 'vehicle', 'savings', 'jewellery', 'land', 'other_house'] as const;

export interface FamilyAnswers {
  fatherHeight: string;
  motherAge: string;
  birthOrder: string;
  under5: string;
  /** null = not asked; [] = asked, owns none of the items. */
  assets: string[] | null;
}

export const emptyFamily: FamilyAnswers = { fatherHeight: '', motherAge: '', birthOrder: '', under5: '', assets: null };

const num = (s: string) => (s.trim() ? Number(s.replace(',', '.')) : null);

/** The API body for the answers: unknown stays null, never 0. */
export function familyBody(a: FamilyAnswers) {
  return {
    father_height_cm: num(a.fatherHeight),
    mother_age_at_birth: num(a.motherAge),
    birth_order: num(a.birthOrder),
    household_under5: num(a.under5),
    household_assets: a.assets,
  };
}

export function familyFrom(c: {
  father_height_cm?: number | null;
  mother_age_at_birth?: number | null;
  birth_order?: number | null;
  household_under5?: number | null;
  household_assets?: string[] | null;
}): FamilyAnswers {
  const s = (v: number | null | undefined) => (v === null || v === undefined ? '' : String(v));
  return {
    fatherHeight: s(c.father_height_cm),
    motherAge: s(c.mother_age_at_birth),
    birthOrder: s(c.birth_order),
    under5: s(c.household_under5),
    assets: c.household_assets ?? null,
  };
}

/** Family questions for the risk model. Every answer is optional. */
export function FamilyQuestions({ value, onChange, heading = true }: { value: FamilyAnswers; onChange: (v: FamilyAnswers) => void; heading?: boolean }) {
  const { t } = useAuth();
  const set = (patch: Partial<FamilyAnswers>) => onChange({ ...value, ...patch });
  const toggle = (k: string) => {
    const cur = value.assets ?? [];
    set({ assets: cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k] });
  };
  const none = value.assets !== null && value.assets.length === 0;
  return (
    <View>
      {heading && <H2 emoji="👪">{t('familyTitle')}</H2>}
      <P muted style={{ fontSize: 13, marginBottom: 12 }}>{t('familyHint')}</P>
      <Field label={t('fatherHeight')} value={value.fatherHeight} onChangeText={(v) => set({ fatherHeight: v })} keyboardType="decimal-pad" placeholder={`${t('eg')} 162`} />
      <Field label={t('motherAgeAtBirth')} value={value.motherAge} onChangeText={(v) => set({ motherAge: v })} keyboardType="number-pad" placeholder={`${t('eg')} 26`} />
      <Field label={t('birthOrder')} value={value.birthOrder} onChangeText={(v) => set({ birthOrder: v })} keyboardType="number-pad" placeholder={`${t('eg')} 2`} />
      <Field label={t('householdUnder5')} value={value.under5} onChangeText={(v) => set({ under5: v })} keyboardType="number-pad" placeholder={`${t('eg')} 1`} />
      <Text style={{ color: colors.text, fontWeight: '700', marginBottom: 6 }}>{t('assetsTitle')}</Text>
      <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
        {HOUSEHOLD_ASSETS.map((k) => (
          <Chip key={k} label={t(`asset_${k}` as TKey)} selected={(value.assets ?? []).includes(k)} onPress={() => toggle(k)} />
        ))}
        <Chip label={t('assetsNone')} selected={none} onPress={() => set({ assets: none ? null : [] })} />
      </View>
    </View>
  );
}
