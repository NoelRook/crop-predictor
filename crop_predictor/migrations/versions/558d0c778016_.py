"""empty message

Revision ID: 558d0c778016
Revises: 76818580eace
Create Date: 2024-11-30 00:41:29.799233

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '558d0c778016'
down_revision = '76818580eace'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('association')
    op.drop_table('challenge')
    op.drop_table('timerecord')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('timerecord',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('elapsed_time', sa.INTEGER(), nullable=True),
    sa.Column('challenge_id', sa.INTEGER(), nullable=True),
    sa.Column('user_id', sa.INTEGER(), nullable=True),
    sa.ForeignKeyConstraint(['challenge_id'], ['challenge.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('challenge',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('question_id', sa.INTEGER(), nullable=True),
    sa.ForeignKeyConstraint(['question_id'], ['question.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('association',
    sa.Column('user_id', sa.INTEGER(), nullable=True),
    sa.Column('challenge_id', sa.INTEGER(), nullable=True),
    sa.ForeignKeyConstraint(['challenge_id'], ['challenge.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], )
    )
    # ### end Alembic commands ###