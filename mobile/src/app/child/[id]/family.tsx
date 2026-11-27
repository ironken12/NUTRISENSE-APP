import { useLocalSearchParams } from 'expo-router';
import React, { useState } from 'react';

import { familyBody, familyFrom, FamilyAnswers, FamilyQuestions } from '../../../components/FamilyQuestions';
import { Button, Card, ErrorBox, Loading, P, Screen } from '../../../components/ui';
import { api, errorText } from '../../../lib/api';
import { useAuth } from '../../../lib/auth';
import type { Child } from '../../../lib/types';
import { useApi } from '../../../lib/useApi';

/** Family details for the risk model (ifls5-lr-2), for children registered before these questions existed. */
export default function FamilyDetails() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { t } = useAuth();
  const child = useApi<Child>(`/api/children/${id}`);
  const [edited, setAnswers] = useState<FamilyAnswers | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  if (!child.data) return <Loading />;
  const answers = edited ?? familyFrom(child.data);

  const save = async () => {
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      await api<Child>(`/api/children/${id}`, { method: 'PATCH', body: familyBody(answers) });
      setSaved(true);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Screen>
      <Card>
        <FamilyQuestions heading={false} value={answers} onChange={(v) => { setAnswers(v); setSaved(false); }} />
      </Card>
      {error && <ErrorBox message={error} />}
      {saved && <P muted>✅ {t('familySaved')}</P>}
      <Button title={t('save')} onPress={save} loading={busy} icon="checkmark" />
    </Screen>
  );
}
