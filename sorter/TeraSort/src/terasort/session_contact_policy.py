"""Versioned exclusion of declared bad contacts from session neighborhoods."""

BAD_CONTACT_POLICY = 'static_bad_exclusion_anchor_pad_v1'


def recorded_contact_exclusion(config):
    value = config.get('bad_contact_neighborhoods')
    if value is None:
        return False
    if value != BAD_CONTACT_POLICY:
        raise ValueError('Unsupported recorded bad-contact neighborhood policy')
    return True
